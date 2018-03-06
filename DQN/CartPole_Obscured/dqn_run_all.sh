#!/bin/bash

for i in {0..9}
do
    python CartPole.py \
--save_path Data/Run_$i \
--training_episodes 5000 \
--training_steps 1000 \
--testing_episodes 100 \
--testing_steps 1000 \
--memory_size 100000 \
--memory_samples 64 \
--start_training_episode 100 \
--target_copy_iterations 0 \
--target_copy_start_steps 0 \
--future_discount 0.99 \
--minimum_epsilon 0.0001 \
--maximum_epsilon 0.9 \
--epsilon_decay 0.9 \
--learning_rate 0.0005 \
--learning_rate_decay 1.0 \
--optimizer adam \
--layer_size_1 512 \
--layer_size_2 256 \
--layer_activation_1 relu \
--layer_activation_2 relu \
--training_goal 400 \
--render_training 0 \
--render_testing 0
done

