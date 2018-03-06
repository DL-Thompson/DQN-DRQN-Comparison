import argparse, os, sys

parser = argparse.ArgumentParser(description='Deep Recurrent Q-Network Arguments')

parser.add_argument('-s', '--save_path', help='Path to Save Files', required=True)
parser.add_argument('-te', '--training_episodes', help='Number of Training Episodes', nargs='?', default=5000, type=int)
parser.add_argument('-ts', '--training_steps', help='Number of Maximum Steps in a Training Episode', nargs='?', default=2000, type=int)
parser.add_argument('-tte', '--testing_episodes', help='Number of Testing Episodes', nargs='?', default=100, type=int)
parser.add_argument('-tts', '--testing_steps', help='Number of Maximum Steps in a Testing Episode', nargs='?', default=2000, type=int)
parser.add_argument('-ms', '--memory_size', help='Maximum Size of Replay Memory.', nargs='?', default=2000, type=int)
parser.add_argument('-mep', '--memory_episodes', help='Number of Episodes to Include in a Memory Sample', nargs='?', default=8, type=int)
parser.add_argument('-mst', '--memory_steps', help='Number of Steps to Include in an Episode Sample', nargs='?', default=8, type=int)
parser.add_argument('-ste', '--start_training_episode', help='Number of Episodes in the Beginning to Choose Random Actions', nargs='?', default=100, type=int)
parser.add_argument('-tc', '--target_copy_iterations', help='Number of Training Iterations Before the Target Network is Updated', nargs='?', default=250, type=int)
parser.add_argument('-tcs', '--target_copy_start_steps', help='Number of Iterations in the First Episode to do Target Copies', nargs='?', default=10, type=int)
parser.add_argument('-fd', '--future_discount', help='Future Discount Rate for Q-Values', nargs='?', default=0.99, type=float)
parser.add_argument('-emi', '--minimum_epsilon', help='Minimum Value of Epsilon for Random Action Selection', nargs='?', default=0.0001, type=float)
parser.add_argument('-emx', '--maximum_epsilon', help='Maximum Value of Epsilon for Random Action Selection', nargs='?', default=0.9, type=float)
parser.add_argument('-ed', '--epsilon_decay', help='Value to Decay Epsilon each Iteration', nargs='?', default=0.9, type=float)
parser.add_argument('-lr', '--learning_rate', help='Agent Learning Rate', nargs='?', default=0.0005, type=float)
parser.add_argument('-lrd', '--learning_rate_decay', help='Value to Decay Learning Rate each Iteration', nargs='?', default=1, type=float)
parser.add_argument('-lrde', '--learning_rate_decay_ep', help='Modulo Episode To do a Decay Calculation', nargs='?', default=500, type=int)
parser.add_argument('-unr', '--lstm_unroll', help='Unroll LSTM for BPTT', nargs='?', default=1, type=int)
parser.add_argument('-opt', '--optimizer', help='Optimizer for BackProp', nargs='?', default='adam', type=str)
parser.add_argument('-ls1', '--layer_size_1', help='Number of Neurons in Dense Layer', nargs='?', default=512, type=int)
parser.add_argument('-ls2', '--layer_size_2', help='Number of Neurons in LSTM Layer', nargs='?', default=256, type=int)
parser.add_argument('-l1a', '--layer_activation_1', help='Activation Function for Layer 1', nargs='?', default='relu', type=str)
parser.add_argument('-l2a', '--layer_activation_2', help='Activation Function for Layer 2', nargs='?', default='relu', type=str)
parser.add_argument('-tg', '--training_goal', help='Average Reward Value to Stop Training', required=True, type=float)
parser.add_argument('-rtr', '--render_training', help='Display the Gym GUI during Training', nargs='?', default=0, type=int)
parser.add_argument('-rts', '--render_testing', help='Display the Gym GUI during Testing', nargs='?', default=0, type=int)
args = parser.parse_args()


config_path = args.save_path + '.config'
if os.path.exists(config_path):
    print 'Naming Prefix has already been used.'
    sys.exit()
file = open(config_path, 'w')
file.write("Save Path: " + str(args.save_path) + '\n')
file.write("Training Episodes: " + str(args.training_episodes) + '\n')
file.write("Training Steps: " + str(args.training_steps) + '\n')
file.write("Testing Episodes: " + str(args.testing_episodes) + '\n')
file.write("Testing Steps: " + str(args.testing_steps) + '\n')
file.write("Memory Size: " + str(args.memory_size) + '\n')
file.write("Memory Episode Samples: " + str(args.memory_episodes) + '\n')
file.write("Memory Step Samples: " + str(args.memory_steps) + '\n')
file.write("Start Training Episode: " + str(args.start_training_episode) + '\n')
file.write("Target Copy Iterations: " + str(args.target_copy_iterations) + '\n')
file.write("Target Copy Start Steps: " + str(args.target_copy_start_steps) + '\n')
file.write("Future Discount: " + str(args.future_discount) + '\n')
file.write("Epsilon Minimum: " + str(args.minimum_epsilon) + '\n')
file.write("Epsilon Maximum: " + str(args.maximum_epsilon) + '\n')
file.write("Epsilon Decay: " + str(args.epsilon_decay) + '\n')
file.write("Learning Rate: " + str(args.learning_rate) + '\n')
file.write("Learning Rate Decay: " + str(args.learning_rate_decay) + '\n')
file.write("Learning Rate Decay Episode: " + str(args.learning_rate_decay_ep) + '\n')
file.write("LSTM Unroll: " + str(args.lstm_unroll) + '\n')
file.write("Optimizer: " + str(args.optimizer) + '\n')
file.write("Layer 1 Size: " + str(args.layer_size_1) + '\n')
file.write("Layer 2 Size: " + str(args.layer_size_2) + '\n')
file.write("Layer 1 Activation: " + str(args.layer_activation_1) + '\n')
file.write("Layer 2 Activation: " + str(args.layer_activation_2) + '\n')
file.write("Training Goal: " + str(args.training_goal) + '\n')
file.close()

def print_args():
    print 'Config Data:'
    print ' Save Path:', args.save_path
    print ' Training Episodes:', args.training_episodes
    print ' Training Steps:', args.training_steps
    print ' Testing Episodes:', args.testing_episodes
    print ' Testing Steps:', args.testing_steps
    print ' Memory Size:', args.memory_size
    print ' Memory Episode Samples:', args.memory_episodes
    print ' Memory Step Samples:', args.memory_steps
    print ' Start Training Episode:', args.start_training_episode
    print ' Target Copy Iterations:', args.target_copy_iterations
    print ' Target Copy Start Steps', args.target_copy_start_steps
    print ' Future Discount:', args.future_discount
    print ' Epsilon Minimum:', args.minimum_epsilon
    print ' Epsilon Maximum:', args.maximum_epsilon
    print ' Epsilon Decay:', args.epsilon_decay
    print ' Learning Rate:', args.learning_rate
    print ' Learning Rate Decay:', args.learning_rate_decay
    print ' Learning Rate Decay Episode:', args.learning_rate_decay_ep
    print ' LSTM Unroll:', args.lstm_unroll
    print ' Optimizer:', args.optimizer
    print ' Layer 1 Size:', args.layer_size_1
    print ' Layer 2 Size:', args.layer_size_2
    print ' Layer 1 Activation:', args.layer_activation_1
    print ' Layer 2 Activation:', args.layer_activation_2
    print ' Training Goal:', args.training_goal
    print ' Render Training:', bool(args.render_training)
    print ' Render Testing:', bool(args.render_testing)
