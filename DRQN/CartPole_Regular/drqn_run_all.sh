#!/bin/bash

for i in {0..9}
do
    python CartPole.py \
--save_path Data/Run_$i \
--training_episodes 5000 \
--training_steps 1000 \
--testing_episodes 100 \
--testing_steps 1000 \
--memory_size 5000 \
--memory_episodes 8 \
--memory_steps 8 \
--start_training_episode 100 \
--target_copy_iterations 250 \
--target_copy_start_steps 0 \
--future_discount 0.99 \
--minimum_epsilon 0.0001 \
--maximum_epsilon 0.9 \
--epsilon_decay 0.9 \
--learning_rate 0.0005 \
--learning_rate_decay 0.8 \
--learning_rate_decay_ep 500 \
--lstm_unroll 1 \
--optimizer adam \
--layer_size_1 512 \
--layer_size_2 256 \
--layer_activation_1 tanh \
--layer_activation_2 tanh \
--training_goal 400 \
--render_training 0 \
--render_testing 0
done

