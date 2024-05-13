import tensorflow as tf
import numpy as np
import scipy.stats as stats
## Actor NN Architecture
import networks as nets
#tf.random.set_seed(1)

## Critic NN Architecture


## DDPG Agent Class (update target, get action, train NN's)
class DDPGAgent:
    def __init__(self, state_dim, action_dim, gamma, tau, actor_lr, critic_lr):
        #variables 

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.actor = nets.Actor(state_dim, action_dim)
        self.critic = nets.Critic(state_dim, action_dim)
        self.target_actor = nets.Actor(state_dim, action_dim)
        self.target_critic = nets.Critic(state_dim, action_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
         # Exploration decay rate
        self.update_target_networks(tau=1.0) # First update, tau set to 1
        

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        #Soft update to target actor w/ tau
        actor_weights = self.actor.get_weights() # retrieve actor weights
        target_actor_weights = self.target_actor.get_weights() # retrieve target actor weights
        for i in range(len(actor_weights)): 
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i] 
        self.target_actor.set_weights(target_actor_weights)

        #Soft update to target critic w/ tau
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    def get_action(self, state, exploration_stddev, greedy):
        state = np.array(state) # convert state to array
         # set exploration floor
        action =self.actor.predict(state.reshape(1, -1))[0] # predict the action
        if greedy:
            return action
        # add exploration noise to action 
        if np.random.rand()<0.5:
            action += exploration_stddev * np.random.randn() 
        else:
            action -= exploration_stddev * np.random.randn() 

        return action
    
    def train(self, replay_buffer, batch_size):
        # grab sample (S-A-R-S') from buffer 
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size) 

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states) #get target action from current target policy
            target_q_values = self.target_critic(next_states, target_actions) #get target Q(S',A)
            y = rewards + (1 - dones) * self.gamma * target_q_values # y = R(S,A) + γQ(S',A)
            
            # Compute the expected future variance
            target_next_actions = self.target_actor(next_states) #Generate target A' w/ target policy pi(S')-> A'
            target_next_q_values = self.target_critic(next_states, target_next_actions) # Generate next target Q with actor

            # Variance at each step: [(Q(S',A') - E[Q(S',A')])^2 ]
            # Note E[.] = tf.reduce_mean, thus below eqn is: E[Var] =E[(Q(S',A') - E[Q(S',A')])**2 ]
            expected_future_variance = tf.reduce_mean(tf.square(target_next_q_values - tf.reduce_mean(target_next_q_values)))
            y -= 0.2*expected_future_variance # Subtract expected future variance from expected future rewards
            
            predicted_q_values = self.critic(states, actions) #We have Q(S',A'), now need Q(S,A)
            critic_loss = tf.reduce_mean( tf.square(y - predicted_q_values)) #L = E[(y-Q(S,A))^2]

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables) # Compute ∇L
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables)) # Optimize Gradient

        with tf.GradientTape() as tape:
            actor_actions = self.actor(states)
            actor_critic_values = self.critic(states, actor_actions)
            entropy = -np.sum(stats.entropy(actor_actions))

            # Define the entropy regularization coefficient
            entropy_coefficient =0.000001
            actor_loss2 = -tf.reduce_mean(actor_critic_values)
            # Modify the actor loss to include entropy regularization
            actor_loss = -tf.reduce_mean(actor_critic_values) + entropy_coefficient * tf.reduce_mean(entropy)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.update_target_networks()
        
        return actor_loss2, critic_loss
