from __future__ import absolute_import, division, print_function

import csv

import matplotlib.pyplot as plt

import tensorflow as tf
import constants
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

from src.rb_environment import ClusterEnv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import numpy as np
tf.compat.v1.enable_v2_behavior()


# Data Collection
def collect_episode(environment, policy, num_episodes, replay_buffer):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


# This loop is so common in RL, that we provide standard implementations of
# these. For more details see the drivers module.

# ***Metrics and Evaluation ***
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        # print('\n\n evaluation started \n')
        while not time_step.is_last():
            action_step = policy.action(time_step)
            # print('action: ', action_step.action)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
        # print('episode return: ', episode_return)

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



def compute_avg_reward(environment, policy, num_episodes=10):
    total_rewards = 0.0
    total_steps = 0

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_rewards = 0.0
        episode_steps = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_rewards += time_step.reward
            episode_steps += 1

        total_rewards += episode_rewards
        total_steps += episode_steps

    avg_reward = total_rewards / total_steps if total_steps > 0 else 0
    return avg_reward.numpy()[0]


# def plot_smoothed_rewards(rewards, window_size=100):
#     smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(smoothed_rewards, label='Reward')
#     plt.ylabel('Reward')
#     plt.xlabel('Step')
#     plt.legend()
#     plt.show()



def train_reinforce(
        # ***Hyperparameters***
        num_iterations=10000,  # @param {type:"integer"}
        collect_episodes_per_iteration=6,  # @param {type:"integer"}
        replay_buffer_max_length=10000,  # @param {type:"integer"}
        fc_layer_params=(100,),
        learning_rate=9e-4,  # @param {type:"number"}
        log_interval=200,  # @param {type:"integer"}
        num_eval_episodes=10,  # @param {type:"integer"}
        eval_interval=1000  # @param {type:"integer"}
):
    file = open('D:\\sumit\\RB_DDRL-master\\output\\avg_returns_'+constants.algo+'_beta_'+str(constants.beta)+'.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return", "AVG_Reward"])

    # *** Environment***
    # 2 environments, 1 for training and 1 for evaluation
    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    # converting pyenv to tfenv
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # ***Agent***

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)

    agent.initialize()

    # *** Policies ***

    # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

    # *** Replay Buffer ***
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # *** Agent Training ***
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    rewards = [avg_reward]
    simple_rewards = [] 

    for _ in range(num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_env, agent.collect_policy, collect_episodes_per_iteration, replay_buffer)

        # Use data from the buffer and update the agent's network.
        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()


        step = agent.train_step_counter.numpy()
        time_step = train_env.current_time_step()

        simple_rewards.append(time_step.reward.numpy()[0])

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            avg_reward = compute_avg_reward(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}, Average Reward = {2}'.format(step, avg_return, avg_reward))
            avg_return_writer.writerow([step, avg_return, avg_reward])
            returns.append(avg_return)
            rewards.append(avg_reward)

    # *** Visualizations ***

    iterations = range(0, num_iterations + 1, eval_interval)
    # plt.plot(iterations, returns, label='Average Return')
    # plt.plot(iterations, rewards, label='Average Reward')
    # plt.ylabel('Average Return / Reward')
    # plt.xlabel('Iterations')
    # plt.legend()
    # plt.show()



    # # Plot the simple rewards with smoothing
    # plot_smoothed_rewards(simple_rewards)