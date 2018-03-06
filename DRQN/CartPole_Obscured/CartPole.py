import DRQN_Parse as parser
from DRQN import DRQN
import gym
from collections import deque

def process_observation(observation):
    return [observation[0], observation[2]]


if __name__ == "__main__":
    print 'Running DRQN Cart Pole...'

    parser.print_args()

    env = gym.make('CartPole-v0')
    num_actions = 2
    num_observations = 2


    drqn = DRQN(parser.args, num_actions, num_observations)

    #Run training episodes
    print 'Running Training Episodes...'
    reward_average = deque(maxlen=100)
    best_average = -float('-inf')
    for episode in range(parser.args.training_episodes):
        total_reward = 0
        observation = env.reset()
        observation = process_observation(observation)
        for step in range(parser.args.training_steps):
            if bool(parser.args.render_training):
                env.render()
            if episode < parser.args.start_training_episode:
                action = drqn.get_random_action()
            else:
                action = drqn.get_action(observation, True)
            next_observation, reward, done, info = env.step(action)
            next_observation = process_observation(next_observation)
            total_reward += reward
            if step == parser.args.training_steps - 1:
                done = True
            drqn.add_transaction(observation, action, next_observation, reward, done)
            if episode > parser.args.start_training_episode:
                drqn.train_model()
            observation = next_observation
            if done:
                break
        drqn.write_training_episode(episode, total_reward)
        reward_average.append(total_reward)
        avg = 0
        if len(reward_average) == 100:
            avg = sum(reward_average) / float(len(reward_average))
            if avg > best_average:
                drqn.save_model('best_avg')
        print 'Episode:', episode, 'Reward:', total_reward, 'Average(100):', avg
        if avg > parser.args.training_goal:
            break
    drqn.save_model('final')

    #Run testing episodes
    print 'Running Testing Episodes...'
    reward_average = deque(maxlen=100)
    for episode in range(parser.args.testing_episodes):
        total_reward = 0
        observation = env.reset()
        observation = process_observation(observation)
        for step in range(parser.args.training_steps):
            if bool(parser.args.render_testing):
                env.render()
            action = drqn.get_action(observation, False)
            next_observation, reward, done, info = env.step(action)
            next_observation = process_observation(next_observation)
            total_reward += reward
            observation = next_observation
            if done:
                break
        drqn.write_testing_episode(episode, total_reward)
        reward_average.append(total_reward)
        avg = 0
        print 'Episode:', episode, 'Reward:', total_reward
        if avg > parser.args.training_goal:
            break
    avg = sum(reward_average) / float(len(reward_average))
    print 'Testing Average:', avg