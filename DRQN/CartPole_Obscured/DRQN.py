import numpy as np
#to reproduce results in keras, the seed must be set before keras is imported
#to do: find some way to make this a parameter
#comment out for random
#np.random.seed(1)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from collections import deque

class DRQN:
    def __init__(self, args, num_actions, num_observations):
        self.args = args
        self.num_actions = num_actions
        self.num_observations = num_observations

        #Create the model that will be trained
        self.model = Sequential()
        self.model.add(Dense(output_dim=self.args.layer_size_1,
                             input_shape=(self.args.memory_steps, self.num_observations),
                             activation=self.args.layer_activation_1))
        self.model.add(LSTM(output_dim=self.args.layer_size_2,
                            activation=self.args.layer_activation_2,
                            unroll=self.args.lstm_unroll))
        self.model.add(Dense(output_dim=self.num_actions, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=self.args.optimizer)

        #Create the model that will calculate target values
        self.model_target = Sequential()
        self.model_target = Sequential.from_config(self.model.get_config())
        self.model_target.set_weights(self.model.get_weights())
        self.model_target.compile(loss='mean_squared_error', optimizer=self.args.optimizer)

        #Since the LSTM has an internal state that is changed when predictions are made,
        #we keep a separate copy of the training model in order to select actions. This
        #is simpler than tracking the internal state and resetting the values
        self.model_action = Sequential()
        self.model_action = Sequential.from_config(self.model.get_config())
        self.model_action.set_weights(self.model.get_weights())
        self.model_action.compile(loss='mean_squared_error', optimizer=self.args.optimizer)

        #Create the Replay Memory
        self.replay_memory = deque(maxlen=self.args.memory_size)
        self.replay_current = []

        self.current_epsilon = self.args.maximum_epsilon
        self.current_learning_rate = self.args.learning_rate

        self.train_iterations = 0
        self.first_iterations = 0
        self.current_episode = 0
        self.current_step = 0

        self.average_train_rewards = deque(maxlen=100)
        self.average_test_rewards = deque(maxlen=100)

        self.train_path = args.save_path + ".train"
        self.train_file = open(self.train_path, 'w')
        self.train_file.write('episode reward average_reward\n')

        self.test_path = args.save_path + ".test"
        self.test_file = open(self.test_path, 'w')
        self.test_file.write('episode reward\n')

        #the lstm layer requires inputs of timesteps
        #a history of of past observations are kept to pass into the lstm
        self.lstm_obs_history = deque([[0 for y in range(self.num_observations)] for x in range(self.args.memory_steps)], self.args.memory_steps)


    def __del__(self):
        self.train_file.close()
        self.test_file.close()
        pass

    def clear_history(self):
        self.lstm_obs_history = deque([[0 for y in range(self.num_observations)] for x in range(self.args.memory_steps)], self.args.memory_steps)


    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_action(self, observation, training):
        #get the current action for the model or a random action depending on arguments
        #and the current episode
        self.lstm_obs_history.append(observation)
        if training and np.random.random() < self.current_epsilon:
            #if training, choose random actions
            return np.random.randint(0, self.num_actions)
        else:
            #choose action from model
            q_values = self.model_action.predict(np.array(self.lstm_obs_history).reshape(1, self.args.memory_steps, self.num_observations))
            max_q = q_values.max(axis=1)
            action_choices = []
            for i, q in enumerate(q_values[0]):
                if q == max_q:
                    action_choices.append(i)
            return np.random.choice(action_choices)


    def add_transaction(self, state, action, next_state, reward, terminal):
        #add a transaction to replay memory, should be called after performing
        #an action and getting an observation
        self.replay_current.append((state, action, next_state, reward, terminal))
        self.current_step += 1
        #make end of episode checks
        if terminal:
            self.replay_memory.append(self.replay_current)
            self.end_of_episode()

    def end_of_episode(self):
        self.current_episode += 1
        self.current_step = 0
        self.clear_history()
        self.replay_current = []
        self.update_action_network()
        self.model_action.reset_states()
        if self.current_epsilon > self.args.minimum_epsilon:
            self.decay_epsilon()
        else:
            self.current_epsilon = self.args.minimum_epsilon
        if self.current_episode % self.args.learning_rate_decay_ep == 0:
            self.decay_learning_rate()

    def sample_memory(self, batch_size, trace_length):
        #samples the replay memory returning a batch_size of random transactions
        sampled_episodes = []
        while True:
            rand_ep = np.random.randint(0, len(self.replay_memory))
            sampled_episodes.append(rand_ep)
            if len(sampled_episodes) == batch_size:
                break
        sampled_traces = []
        for ep in sampled_episodes:
            episode = self.replay_memory[ep]
            start_step = np.random.randint(0, max(1, len(episode) - trace_length + 1))
            current_trace = episode[start_step:start_step + trace_length]
            action = current_trace[-1][1]
            reward = current_trace[-1][3]
            terminal = current_trace[-1][4]
            states = []
            next_states = []
            for step, transaction in enumerate(current_trace):
                states.append(transaction[0])
                next_states.append(transaction[2])
            if len(current_trace) < trace_length:
                empty = [0 for x in states[0]]
                for i in range(trace_length - len(current_trace)):
                    states.insert(0, empty)
                    next_states.insert(0, empty)
            sampled_traces.append([states, action, next_states, reward, terminal])
        return sampled_traces

    def train_model(self):
        if len(self.replay_memory) < self.args.memory_episodes:
            print 'Not enough transactions in replay memory to train.'
            return
        if self.train_iterations >= self.args.target_copy_iterations:
            self.update_target_network()
        if self.first_iterations < self.args.target_copy_start_steps:
            # update the target network a few times on episode 0 so
            # the model isn't training toward a completely random network
            self.update_target_network()
            self.first_iterations += 1

        self.model.reset_states()
        self.model_target.reset_states()

        samples = self.sample_memory(self.args.memory_episodes, self.args.memory_steps)
        observations = next_observations = rewards = np.array([])
        actions = terminals = np.array([], dtype=int)
        for transaction in samples:
            observations = np.append(observations, transaction[0])
            actions = np.append(actions, transaction[1])
            next_observations = np.append(next_observations, transaction[2])
            rewards = np.append(rewards, transaction[3])
            terminals = np.append(terminals, transaction[4])
        observations = observations.reshape(self.args.memory_episodes, self.args.memory_steps, self.num_observations)
        next_observations = next_observations.reshape(self.args.memory_episodes, self.args.memory_steps,  self.num_observations)
        targets = updates = None
        if self.args.target_copy_iterations == 0:
            #this instance is not using a target copy network, use original model
            targets = self.model.predict(observations)
            updates = rewards + (1. - terminals) * self.args.future_discount * self.model.predict(next_observations).max(axis=1)
        else:
            #this instance uses a target copy network
            targets = self.model_target.predict(observations)
            updates = rewards + (1. - terminals) * self.args.future_discount * self.model_target.predict(next_observations).max(axis=1)
        for i, action in enumerate(actions):
            targets[i][action] = updates[i]
        self.model.fit(observations, targets, nb_epoch=1, batch_size=self.args.memory_episodes, verbose=0)

        self.train_iterations += 1

    def update_target_network(self):
        self.model_target.set_weights(self.model.get_weights())

    def update_action_network(self):
        self.model_action.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        self.current_epsilon *= self.args.epsilon_decay

    def decay_learning_rate(self):
        self.current_learning_rate *= self.args.learning_rate_decay

    def write_training_episode(self, episode, reward):
        self.average_train_rewards.append(reward)
        self.train_file.write(str(episode) + ' ')
        self.train_file.write(str(reward) + ' ')
        if len(self.average_train_rewards) >= 100:
            self.train_file.write(str(np.mean(self.average_train_rewards)))
        self.train_file.write('\n')

    def write_testing_episode(self, episode, reward):
        self.average_test_rewards.append(reward)
        self.test_file.write(str(episode) + ' ')
        self.test_file.write(str(reward) + ' ')
        self.test_file.write('\n')

    def save_model(self, file_name):
        file_path = self.args.save_path + '_' + file_name + '.model'
        self.model.save(file_path, True)