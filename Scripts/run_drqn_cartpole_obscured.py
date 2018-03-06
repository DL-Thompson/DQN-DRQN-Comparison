import argparse
parser = argparse.ArgumentParser(description='Run OpenAI Gym Model')
parser.add_argument('model_path')
parser.add_argument('save_file')
parser.add_argument('-te', '--test_episodes', help='Number of test episodes', nargs='?', default=100, type=int)
parser.add_argument('-ts', '--test_steps', help='Maximum number of test steps', nargs='?', default=1000, type=int)
parser.add_argument('-r', '--render', help='Render graphics', nargs='?', default=0, type=int)
parser.add_argument('-ms', '--memory_steps', help='Number of trace length steps used to train', nargs='?', default=0, type=int)
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


print 'Running DRQN Cart Pole Partially Observable Tests...'
print 'Using Model: ', args.model_path

model = keras.models.load_model(args.model_path)
model.reset_states()
print model.get_config()

env = gym.make('CartPole-v0')

state_history = deque([[0 for y in range(NUM_OBSERVATIONS)] for x in range(args.memory_steps)], args.memory_steps)
def clear_state_history():
    global state_history
    state_history = deque([[0 for y in range(NUM_OBSERVATIONS)] for x in range(args.memory_steps)], args.memory_steps)

def get_action(observation):
    global state_history
    state_history.append(observation)
    observation = np.array([state_history]).reshape((1, args.memory_steps, NUM_OBSERVATIONS))
    action_values = model.predict(observation)
    max_action = np.argmax(action_values)
    return max_action

def run_test_episode():
    model.reset_states()
    clear_state_history()
    total_reward = 0
    observation = env.reset()
    observation = process_observation(observation)
    done = False
    for step in range(args.test_steps):
        if done:
            break
        if bool(args.render):
            env.render()
        action = get_action(observation)
        next_observation, reward, done, info = env.step(action)
        next_observation = process_observation(next_observation)
        total_reward += reward
        observation = copy.copy(next_observation)
    return total_reward

print 'Running Test Episodes...'
file = open(str(args.save_file), 'w')
file.write('episode reward\n')
reward_queue = deque(maxlen=args.test_episodes)
for i in range(args.test_episodes):
    reward = run_test_episode()
    reward_queue.append(reward)
    print 'Test Episode: ', i, 'Reward: ', reward, ' Avg: ', np.mean(reward_queue)
    file.write(str(i) + " ")
    file.write(str(reward))
    file.write("\n")
    file.flush()
file.close()