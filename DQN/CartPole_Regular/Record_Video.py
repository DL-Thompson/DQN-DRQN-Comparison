import DQN_Parse as parser
from DQN import DQN
import gym
from collections import deque
from gym import wrappers

def process_observation(observation):
    return observation

def record_all_ep(ep):
    return not ep % 10

if __name__ == "__main__":
    print 'Running DQN Cart Pole...'

    parser.print_args()

    env = gym.make('CartPole-v1')
    env = wrappers.Monitor(env, '../../Video_CartPole', record_all_ep)
    num_actions = 2
    num_observations = 4


    dqn = DQN(parser.args, num_actions, num_observations)

    #Run training episodes
    print 'Recording Training Episodes...'
    reward_average = deque(maxlen=100)
    best_average = 0
    for episode in range(parser.args.training_episodes):
        total_reward = 0
        observation = env.reset()
        observation = process_observation(observation)
        for step in range(parser.args.training_steps):
            if bool(parser.args.render_training):
                env.render()
            if episode < parser.args.start_training_episode:
                action = dqn.get_random_action()
            else:
                action = dqn.get_action(observation, True)
            next_observation, reward, done, info = env.step(action)
            next_observation = process_observation(next_observation)
            total_reward += reward
            if step == parser.args.training_steps - 1:
                done = True
            dqn.add_transaction(observation, action, next_observation, reward, done)
            if episode > parser.args.start_training_episode:
                dqn.train_model()
            observation = next_observation
            if done:
                break
        dqn.write_training_episode(episode, total_reward)
        reward_average.append(total_reward)
        avg = 0
        if len(reward_average) == 100:
            avg = sum(reward_average) / float(len(reward_average))
            if avg > best_average:
                dqn.save_model('best_avg')
        print 'Episode:', episode, 'Reward:', total_reward, 'Average(100):', avg
        if avg > parser.args.training_goal:
            break
    dqn.save_model('final')


