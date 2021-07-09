import copy
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from keras_radam import RAdam
from modifiedTensorboard import ModifiedTensorBoard
import time


class actor_model:
    def __init__(self, input_shape, n_actions, lr, optimizer):
        x_input = tf.keras.Input(input_shape)
        self.n_actions = n_actions
        # define hidden layers
        x = tf.keras.layers.Dense(512, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
            x_input)
        x = tf.keras.layers.Dense(256, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
            x)
        x = tf.keras.layers.Dense(64, activation="elu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
            x)
        output_dist = tf.keras.layers.Dense(self.n_actions, activation="softmax")(x)

        self.actor = tf.keras.models.Model(inputs=x_input, outputs=output_dist)
        self.actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.n_actions], y_true[:,
                                                                                                1 + self.n_actions:]
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.backend.clip(prob, 1e-10, 1.0)
        old_prob = K.backend.clip(old_prob, 1e-10, 1.0)

        ratio = K.backend.exp(K.backend.log(prob) - K.backend.log(old_prob))

        p1 = ratio * advantages
        p2 = K.backend.clip(ratio, min_value=1 - 0.2, max_value=1 + 0.2) * advantages

        actor_loss = -K.backend.mean(K.backend.minimum(p1, p2))

        entropy = -(y_pred * K.backend.log(y_pred + 1e-10))
        entropy = 0.001 * K.backend.mean(entropy)
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.actor.predict(state)


class critic_Model:
    def __init__(self, input_shape, n_actions, lr, optimizer):
        x_input = tf.keras.layers.Input(input_shape)
        old_values = tf.keras.layers.Input(shape=(1,))

        x = tf.keras.layers.Dense(512, activation="elu", kernel_initializer='he_uniform')(x_input)
        x = tf.keras.layers.Dense(256, activation="elu", kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.Dense(64, activation="elu", kernel_initializer='he_uniform')(x)
        value = tf.keras.layers.Dense(1, activation=None)(x)

        self.critic = tf.keras.models.Model(inputs=[x_input, old_values], outputs=value)
        self.critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            clipped_value_loss = values + K.backend.clip(y_pred - values, -0.2, 0.2)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2

            value_loss = 0.5 * K.backend.mean(K.backend.maximum(v_loss1, v_loss2))
            return value_loss

        return loss

    def predict(self, state):
        return self.critic.predict([state, np.zeros((state.shape[0], 1))])


class PPOAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.lr = 1e-4  # 0.0001
        self.EPISODES = 10000
        self.eps = 0
        self.epochs = 0
        self.shuffle = False
        self.Training_batch = 1000
        self.optimizer = K.optimizers.Adam
        self.AGGREGATE_STATS_EVERY = 1  # stats the reward for 1 episodes
        self.replay_count = 0
        self.MODEL_NAME = "model"
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(self.MODEL_NAME, int(time.time())))
        # create model
        self.actor = actor_model(input_shape=self.state_size, n_actions=self.n_actions, lr=self.lr,
                                 optimizer=self.optimizer)
        self.critic = critic_Model(input_shape=self.state_size, n_actions=self.n_actions, lr=self.lr,
                                   optimizer=self.optimizer)

        # save model
        self.actor_name = f"PPO_Actor.h5"
        self.critic_name = f"PPO_Critic.h5"

    def act(self, state):
        actions_dist = self.actor.predict(state)[0]  # bcz it is a tensor
        action_random_choice = np.random.choice(self.n_actions, p=actions_dist)
        action_onehot = np.zeros([self.n_actions])
        action_onehot[action_random_choice] = 1
        return action_random_choice, actions_dist, action_onehot

    def discouted_reward(self, reward):
        gamma = 0.99
        running_add = 0
        discouted_r = np.zeros_like(reward)
        for i in reversed(range(len(reward))):
            running_add = running_add * gamma + reward[i]
            discouted_r[i] = running_add
        discouted_r -= np.mean(discouted_r)
        discouted_r /= (np.std(discouted_r) + 1e-8)
        return discouted_r

    def get_gae(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * n - v for r, d, n, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for i in reversed(range(len(rewards))):
            gaes[i] = gaes[i] + (1 - dones[i]) * gamma * lamda * gaes[i]
        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        advantages, target = self.get_gae(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.actor.actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.critic.critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.replay_count += 1

    def load(self):
        self.actor.actor.load_weights(self.actor_name)
        self.critic.critic.load_weights(self.critic_name)

    def save(self):
        self.actor.actor.save_weights(self.actor_name)
        self.critic.critic.save_weights(self.critic_name)

    def compute_entropy(self, prediction):
        entropy = 0
        for i in range(self.n_actions):
            entropy -= prediction[i] * np.log(prediction[i])
        return entropy

    def run(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score = False, 0
        while True:
            states, next_states, actions, rewards, actions_dist, dones, mean_entropy = [], [], [], [], [], [], []
            while not done:
                self.env.render()
                action, prediction, action_onehot = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                entropy = self.compute_entropy(prediction)
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                actions_dist.append(prediction)
                mean_entropy.append(entropy)
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                print("Prob Distribution : ", prediction, " Reward : ", reward, " Entropy : ", entropy)
                if done:
                    self.eps += 1
                    self.replay(states, actions, rewards, actions_dist, dones, next_states)
                    # save weights of model
                    self.save()
                    # after each episode we re-trained
                    state, done, score = self.env.reset(), False, 0
                    state = np.reshape(state, [1, self.state_size[0]])
                    average_reward = sum(rewards) / len(
                        rewards)
                    min_reward = min(rewards)
                    max_reward = max(rewards)
                    average_entropy = sum(mean_entropy) / len(
                        mean_entropy)
                    self.tensorboard.step = self.eps
                    self.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward,
                                                  reward_max=max_reward, entropy=average_entropy)
            if self.eps >= self.EPISODES:
                break
        self.env.close()

    def test(self, episodes=100):
        self.load()
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.actor.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    break
        self.env.close()


if __name__ == "__main__":
    env_name = "LunarLander-v2"
    agent = PPOAgent(env_name)
    agent.run()
    # agent.test()
