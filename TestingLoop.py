import tensorflow as tf
import os 
import gym
import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import csv
import matplotlib.pyplot as plt
import hedging_envs as h_envs
import csv

class TestingLoop():
        def __init__(self, symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price, N,  tc,prices, path, real):            
                    self.day = day
                    self.rho = rho
                    self.nu = nu
                    self.tc = tc
                    
                    self.strike_price = strike
                    self.vol = volatility
                    self.initial_vol = volatility
                    self.N = N
                    self.sabr = sabr
                    self.symbol = symbol
                    self.real = real
                    self.prices = prices
                    self.path = path
                    self.T = day/365 ## 365 - bloomberg lists calendar days
                    self.kappa = 0.005
                    dt = self.T / N
                    r = 0.05
                    if self.real:
                         episodes = 1
                         self.initial_stock_price = self.prices[0]

                    else:
                        self.initial_stock_price = initial_stock_price
                        episodes =10000
                   
                    if self.sabr:
                        env = h_envs.OptionHedgingEnv_svol(strike_price=strike,initial_stock_price=initial_stock_price,  risk_free_rate=r,
                                volatility=volatility, time_horizon=self.T,  max_steps= N, rho=rho,nu=nu, day = day, prices = prices, symbol = symbol, real = self.real, kappa = 0.005)
                    else:        
                        env = h_envs.OptionHedgingEnv_GBM(strike_price=self.strike_price, initial_stock_price=self.strike_price, risk_free_rate=0.05, 
                                            volatility=self.vol, time_horizon=self.T, kappa = 0.005, max_steps=N)

                    loaded_actor = tf.saved_model.load(self.path)           
                    def retrieve_action(state):
                        #print(state)
                        state_array = np.reshape(np.array(state), (1, 3))
                        tensor_state = tf.convert_to_tensor(state_array, dtype=tf.float32)


                        RL_action= loaded_actor(tensor_state)

                        #print(RL_action.numpy()[0][0])
                        return -RL_action.numpy()[0][0]
                    import csv
                    def load_exercise_boundary_from_csv(filename):
                        time_points = []
                        exercise_boundary = []
                        with open(filename, 'r', newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                time_points.append(float(row['Time']))
                                if self.sabr:
                                    exercise_boundary.append(float(row['Exercise Boundary']))
                                else:
                                     exercise_boundary.append(float(row['bin']))
                                     
                        return time_points, exercise_boundary


                    # Load the exercise boundary data
                    if self.sabr:
                         if self.symbol is None:
                              exercise_boundary_filename = 'cheb_fit2.csv'  # Adjust the filename if necessary
                         else:
                                  
                            exercise_boundary_filename = f'{symbol}_fit{strike}-{day}.csv'  # Adjust the filename if necessary
                    else:
                         exercise_boundary_filename = 'binomial.csv'

                    time_points, exercise_boundary = load_exercise_boundary_from_csv(exercise_boundary_filename)
                    #print(exercise_boundary)
                    #print(time_points)
                    
                    new_time_points = np.linspace(0, self.T, N)
                    #print(new_time_points)
                    # Interpolate the exercise boundary values at the new time points
                    exercise_boundary = np.interp(new_time_points, time_points, exercise_boundary)


                    #print(exercise_boundary)






                    final_port_values_RL = []

                    final_port_values_d = []

                    actor_batchnorm = tf.keras.layers.BatchNormalization()


                    RL_actions=[]
                    deltas = []
                    early_exercised_count = 0
                    for i in range(episodes):
                        # Initialize
                        t = 0
                        state = env.reset()
                        stock_price = state[0]
                        stock_path = [stock_price]
                        time_to_maturity = self.T - t
                        # Get first actions
                        #option_priceA = env.option_prices[0][0] ## activate this if you want to compare against alpha hedge
                        option_priceA, delta= env.get_option_price(time_to_maturity, stock_price)
                        if self.sabr:
                            option_price  = env.sim_option_price(stock_price, time_to_maturity, self.initial_vol, 0) ## RL
                        else:
                             
                            option_price = env.option_prices[0][0]
                        action = retrieve_action(env.normalize_state(state))
                        batch_normalized_action = actor_batchnorm(tf.convert_to_tensor([action], dtype=tf.float32))
                        RL_action_old = tf.squeeze(batch_normalized_action).numpy()

                        
                    
                        
                        option_exercised = False
                        exercised_flags = []
                        bank_account_d =-delta*stock_price + option_priceA
                        bank_account_RL = (option_price - RL_action_old * stock_price)
                        
                        
                        for j in range(N - 1):
                            # Take step
                            t += dt
                            time_to_maturity = self.T - t
                            state, _, done, _= env.step([RL_action_old])
                            stock_path.append(state[0])
                            new_stock_price = state[0]
                            _, new_delta= env.get_option_price(time_to_maturity, new_stock_price)
                            #new_delta = env.interpolate_alpha(state[0]) ## alpha hedge
                            

                            # Form state and get action
                            state = np.array([new_stock_price, time_to_maturity, RL_action_old])
                        
                            action = retrieve_action(env.normalize_state(state))
                            batch_normalized_action = actor_batchnorm(tf.convert_to_tensor([action], dtype=tf.float32))
                            RL_action = tf.squeeze(batch_normalized_action).numpy()
                            
                        
                            # Check if the stock price crosses the exercise boundary

                            if not option_exercised and new_stock_price <= exercise_boundary[j]:
                                # Exercise the American put option
                                option_exercised = True
                                exercised_flags.append(True)  # Record early exercise
                                early_exercised_count += 1  # Increment the count of early exercised options
                                final_pnl_RL = bank_account_RL*np.exp(r*dt) - max(0, strike - new_stock_price)
                                final_pnl_RL += RL_action_old * new_stock_price
                                final_pnl_RL*= np.exp(-r*(j*dt))
                                final_pnl_d = bank_account_d*np.exp(r*dt) - max(0, strike - new_stock_price) 
                                final_pnl_d += delta * new_stock_price
                                final_pnl_d*= np.exp(-r*(j*dt))
                                final_port_values_RL.append(final_pnl_RL)
                                final_port_values_d.append(final_pnl_d)
                                
                                
                                break
                            bad = bank_account_d
                            bar = bank_account_RL   
                            bank_account_d = bank_account_d*np.exp(r*dt) + (delta - new_delta)*new_stock_price - 0.03*np.abs(delta-new_delta)*new_stock_price
                            bank_account_RL = bank_account_RL*np.exp(r*dt) + (RL_action_old -RL_action  )*new_stock_price -0.03*np.abs(RL_action_old - RL_action)*new_stock_price
                        

                            
                            stock_price = new_stock_price
                            DlT = delta
                            delta = new_delta
                            RlT = RL_action_old
                            RL_action_old = RL_action
                            
                        
                        if i % 250 == 0:
                                print(i)
                        if not option_exercised:
                        # Calculate final PnL as the difference between invested portfolio and short stock position
                            
                            final_pnl_RL = bar*np.exp(r*dt) - max(0, strike- stock_price)
                            final_pnl_RL +=  RlT* stock_price
                            final_pnl_RL*= np.exp(-r)
                            final_pnl_d = bad*np.exp(r*dt) - max(0, strike - stock_price) 
                            final_pnl_d += DlT* stock_price
                            final_pnl_d *= np.exp(-r)
                        

                            final_port_values_RL.append(final_pnl_RL)
                            final_port_values_d.append(final_pnl_d)
                            


                    #%%
                    if self.symbol is None:
                         csv_file_path = f'sim_backtest_3tc'
                    else:            
                        csv_file_path = f'sim_backtest_{symbol}-{strike}-{day}3tc'

        # Combine arrays into a list of rows
                    data2 = list(zip(final_port_values_RL, final_port_values_d))

                    # Write to CSV file
                    with open(csv_file_path, 'w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                    
                        # Write header if needed
                        writer.writerow(['RL', 'Delta'])
                    
                        # Write data rows
                        writer.writerows(data2)
              
                    
            

                    rl_mean = np.mean(final_port_values_RL)
                    rl_std = np.std(final_port_values_RL)
                    delta_mean = np.mean(final_port_values_d)
                    delta_std = np.std(final_port_values_d)

                    print(f"K = {strike}, Mat = {day}, RL Mean: ", rl_mean)
                    print(f"K = {strike}, Mat = {day},Alp Mean: ", delta_mean)
                    print(f"K = {strike}, Mat = {day},RL Var: ", rl_std)
                    print(f"K = {strike}, Mat = {day},Delta Var: ", delta_std)