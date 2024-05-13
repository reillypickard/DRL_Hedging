import tensorflow as tf
import os 
import numpy as np
from scipy.stats import norm
import hedging_envs as h_envs
import DDPGAgent as ddpg


os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
tf.keras.utils.disable_interactive_logging() 
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Num GPUs Available:", len(gpus))
        print("Num Logical GPUs Available:", len(logical_gpus))
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


#%% Main training loop
class TrainingLoop:
    def __init__(self, symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price):
    # Define hyperparameters and create environment
        self.state_dim = 3
        self.action_dim = 1
        ### Can loop over these for the hyperparam experiments
        self.max_episodes =5000
        self.max_steps_per_episode = 25
        self.actor_lr = 0.5e-5
        self.critic_lr = 0.5e-3
        #######

        self.batch_size = 256
        self.gamma  =1
        self.tau = 0.00005
        self.sabr = sabr
        self.symbol =symbol
        if self.symbol is None:
            self.initial_stock_price =100
            self.time_horizon=1
            self.day=365 ## calendar days, always divide by 365 to get time horiz in yrs. (bloomberg lists maturity in calendar days)
            self.rho = -0.4
            self.nu = 0.1
            self.volatility = 0.2
            self.strike_price =100
        else:
            self.strike_price = strike
            self.rho = rho
            self.nu = nu
            self.volatility = volatility
            self.day= day
            self.time_horizon = day/365
            self.initial_stock_price = initial_stock_price
            


        # Apply batch normalization to actor network
        actor_batchnorm = tf.keras.layers.BatchNormalization()
            ### Specify environment (sabr or GBM)
            ## GBM Args: strike_price, initial_stock_price, risk_free_rate, volatility, time_horizon, max_steps, kappa
            ## SABR Args: strike_price, initial_stock_price, risk_free_rate, volatility (inititial), time_horizon, max_steps,  rho, nu,day, symbol, kappa, real
            ### if not doing a market calibrated agent, symbol = None, manually enter rho, nu
        if self.sabr:
                env = h_envs.OptionHedgingEnv_svol(strike_price=self.strike_price, initial_stock_price=self.strike_price, risk_free_rate=0.05, 
                                            volatility=self.volatility, time_horizon=self.time_horizon, max_steps=25,  rho=self.rho, nu=self.nu,day=self.day, symbol=self.symbol, 
                                            real = False, kappa = 0.005, prices = None)
        else:
                env = h_envs.OptionHedgingEnv_GBM(strike_price=self.strike_price, initial_stock_price=self.strike_price, risk_free_rate=0.05, 
                                            volatility=self.volatility, time_horizon=self.time_horizon, max_steps=25, kappa = 0.005)
        agent = ddpg.DDPGAgent(state_dim=self.state_dim, action_dim=self.action_dim, gamma=self.gamma, tau=self.tau, actor_lr=self.actor_lr, critic_lr=self.critic_lr)

        class ReplayBuffer:
                def __init__(self, buffer_size=10000):
                    self.buffer_size = buffer_size
                    self.buffer = []
                    self.buffer_idx = 0
            
                def add(self, experience):
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.append(None)
                    self.buffer[self.buffer_idx] = experience
                    self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

                def sample(self, batch_size):
                    if len(self.buffer) < batch_size:
                        batch = np.arange(len(self.buffer))  # Sample all available experiences
                    else:
                        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
                    states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in batch])
                    return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states), np.array(dones, dtype=np.float32)

        replay_buffer = ReplayBuffer()
        episode_rewards = []
            # Add noise injection for exploration
        SD = 3 # Iniitial exploration
        final_SD = 0 # Final exploration
        SD_decay = 0.99993
        exploration_stddev = SD
        actor_losses = []
        critic_losses = []
        for episode in range(self.max_episodes):
                state = env.reset()
                T =self.time_horizon
                dt = T/self.max_steps_per_episode
                r = 0.05
                episode_reward = 0.0
                t =0
                
                for step in range(self.max_steps_per_episode):
                    t+= dt
                    time_to_maturity = T-t
                    normalized_state = env.normalize_state(state)  # Normalize the current state
                    action = agent.get_action(normalized_state, exploration_stddev, greedy = False)
                    action = np.clip(action, 0 ,1)
                    batch_normalized_action = actor_batchnorm(tf.convert_to_tensor([action], dtype=tf.float32))
                    action = tf.squeeze(batch_normalized_action).numpy()
                    next_state, reward, done, _ = env.step([action])
                    replay_buffer.add((state, [action], [reward], env.normalize_state(next_state), [done]))
                    episode_reward += reward
                    actor_loss, critic_loss = agent.train(replay_buffer, batch_size=self.batch_size)
                    state = env.normalize_state(next_state)
                
                SD *= SD_decay # decay the exploration on each call
                exploration_stddev = max(final_SD, SD )
                episode_rewards.append(episode_reward)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                if episode%250 == 0:
                                print(f"Episode {episode + 1}/{self.max_episodes}, Episode Reward: {episode_reward:.2f}")

        print("Training finished.")


            #%%
            # Save the actor and critic modmcels using SavedModel format
        if symbol == None:
            actor_model_path = 'agent' ## and add other hyperparams you want
        else:
             actor_model_path = f'{self.symbol}{self.strike}-{self.day}'
             
        critic_model_path = 'critic_model'

        tf.saved_model.save(agent.actor, actor_model_path)
        tf.saved_model.save(agent.critic, critic_model_path)

        print("Trained models saved.")