
import tensorflow as tf
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(action_dim, activation='sigmoid')

    def call(self, state):
        
        x = self.fc1(state)
        x = self.fc2(x)
        action = self.fc3(x)
        return action
        
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
       
   
     
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(1)
     
    def call(self, state, action):
        
        x = self.fc1(state)
        x = tf.concat([x, action], axis=-1)
        x = self.fc2(x)
        q_value = self.fc3(x)
        return q_value
       