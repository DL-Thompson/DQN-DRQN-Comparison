import argparse
parser = argparse.ArgumentParser(description='Run OpenAI Gym Model')
parser.add_argument('model_path')
parser.add_argument('save_file')
parser.add_argument('-te', '--test_episodes', help='Number of test episodes', nargs='?', default=100, type=int)
parser.add_argument('-ts', '--test_steps', help='Maximum number of test steps', nargs='?', default=1000, type=int)
parser.add_argument('-r', '--render', help='Render graphics', nargs='?', default=0, type=int)
args = parser.parse_args()

import keras
import gym
from collections import deque
import copy
import numpy as np

NUM_OBSERVATIONS = 2
NUM_ACTIONS = 2

def process_observation(observation):
    return [observation[0], observation[2]]


print 'Running DQN Cart Pole Partially Observable Tests...'
print 'Using Model: ', args.model_path

model = keras.models.load_model(args.model_path)
print model.get_config()

env = gym.make('CartPole-v0')

def get_action(model, observation, epsilon):
    observation = np.array([observation])
    action_values = model.predict(observation)
    assert len(action_values[0]) == NUM_ACTIONS, 'Error: Action Size Mis-Match'
    action_probability = np.ones(NUM_ACTIONS) * epsilon / NUM_ACTIONS
    max_action = np.argmax(action_values)
    action_probability[max_action] += 1.0 - epsilon
    return np.random.choice(NUM_ACTIONS, p=action_probability)

def run_test_episodes(env, test_steps, model):
    total_reward = 0
    observation = env.reset()
    observation = process_observation(observation)
    done = False
    for step in range(test_steps):
        if done:
            break
        if bool(args.render):
            env.render()
        action = get_action(model, observation, 0)
        next_observation, reward, done, info = env.step(action)
        next_observation = process_observation(next_observation)
        total_reward += reward
        observation = copy.copy(next_observation)
    return total_reward


file = open(str(args.save_file), 'w')
file.write('episode reward\n')
reward_queue = deque(maxlen=args.test_episodes)
for i in range(args.test_episodes):
    reward = run_test_episodes(env, args.test_steps, model)
    reward_queue.append(reward)
    print 'Test Episode: ', i, 'Reward: ', reward, ' Avg: ', np.mean(reward_queue)
    file.write(str(i) + " ")
    file.write(str(reward))
    file.write("\n")
    file.flush()
file.close()