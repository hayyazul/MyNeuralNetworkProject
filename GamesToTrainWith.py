import os
import pickle

import gymnasium as gym
import random
import numpy as np

import QLearningAgent as QLA
import NeuralNetwork as NN
import NN_Visualizations as NNV

"""
Trains Q-Agent on Lunar Lander.

Details:
The Q-Agent NN should take in 8 values as input and output an array of size 5(largest index 4).
"""

def episode(AI: QLA.QAgent, explore_rate=0.05, diagnostics=True):
    """
    Gathers training data for the AI.
    :param AI: A Q-Learning agent.
    :param explore_rate: Probability of choosing a random action.
    :param diagnostics: Bool of whether to record diagnostic variables.
    :return: A list containing tuples with two items, the first an array of the observation, the second an array of
    what the expected return values should be.
    """
    q_values = None
    last_q_values = None

    training_data = []
    env = gym.make(ENVIRONMENT)
    observation, info = env.reset()
    last_observation = observation

    episode_begin = True
    tick = 0
    while True:
        tick += 1
        if episode_begin:
            q_values = AI.get_q_values(observation)
            last_q_values = q_values
            episode_begin = False

        # Take an action
        if random.random() < explore_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_values)

        # Then iterate the state
        observation, reward, terminated, truncated, info = env.step(action)
        pole_angle = observation[2]
        reward = -abs(pole_angle) + tick / 10000

        # Play/gather data until the game ends.
        if terminated:
            # Diagnostic var.
            if diagnostics:
                AVERAGE_REWARD.append(tick)
            # Same as with the game not ending, but without the need to evaluate future rewards as none will exist.
            desired_output = AI.update_rule(-2, 0, action, last_q_values)
            training_data.append((last_observation, desired_output))
            break

        elif tick > 1000:
            # Diagnostic var.
            if diagnostics:
                AVERAGE_REWARD.append(tick)
            # Same as with the game not ending, but without the need to evaluate future rewards as none will exist.
            desired_output = AI.update_rule(2, 0, action, last_q_values)
            training_data.append((last_observation, desired_output))
            break

        else:
            # AI stuff, calculates the q_value for the current observation
            q_values = AI.get_q_values(observation)
            max_q_value = np.max(q_values)

            # Append to training data both the previous state and the desired outputs.
            desired_output = AI.update_rule(reward, max_q_value, action, last_q_values)
            training_data.append((last_observation, desired_output))

            last_q_values = q_values
            last_observation = observation

    return training_data


def visualize_ai(AI, diagnostics=False, render=True, terminal=True):
    """
    Lets the AI play the lunar lander game.
    :param terminal:
    :param AI:
    :param diagnostics: Whether to record diagnostic vars.
    :param render: Whether to render the env.
    :return:
    """
    if render:
        env = gym.make(ENVIRONMENT, render_mode="human")
    else:
        env = gym.make(ENVIRONMENT)

    observation, info = env.reset()

    tick = 0
    while True:
        tick += 1
        # Get Q-Values
        q_values = AI.get_q_values(observation)

        # Take an action
        action = np.argmax(q_values)

        # Then iterate the state
        reward = 0
        observation, reward, terminated, truncated, info = env.step(action)
        reward = -abs(observation[2])
        '''if diagnostics:
            AVERAGE_REWARD.append(reward)'''

        # Play until the game ends.
        if not terminal and tick % 100 == 0:
            print(tick)

        if (terminated or truncated) and terminal:
            break


# --- Test Vars --- #

def save_ai(ai, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'wb') as file:
        pickle.dump(ai, file)


random.seed(123)
np.random.seed(123)

training_set = []
AVERAGE_REWARD = []  # List of rewards, not a list of average rewards. Bad naming.

ENVIRONMENT = 'CartPole-v1'

if __name__ == "__main__":
    # Code to find out size of input/output.
    env0 = gym.make(ENVIRONMENT)
    obs, inf = env0.reset(seed=0)
    input_size = len(obs)
    output_size = env0.action_space.n

    test_AI = QLA.QAgent((NN.NeuralNetworkLayer(input_size, 16, activation_function='relu', disable_bias=True),
                          NN.NeuralNetworkLayer(16, 8, activation_function='relu', disable_bias=True),
                          NN.NeuralNetworkLayer(8, output_size, activation_function='tanh', disable_bias=True)),
                         alpha=0.5, discount=0.99, learning_rate=0.01)

    """
    It learns!!!! (sort of)
    """

    explr_rate = 0.1
    for __ in range(100001):
        if __ == 5000:
            explr_rate = 0.02
            test_AI.alpha = 0.1

        if __ == 10000:
            explr_rate = 0.01
            test_AI.alpha = 0.05

        if __ == 40000:
            explr_rate = 0.005
            test_AI.alpha = 0.02

        for _ in range(1):
            training_set.extend(episode(test_AI, explore_rate=explr_rate))

        if __ % 1000 == 0:
            print(f"Dataset Generation Finished! No. {__}")
            print(f"Average Reward: {sum(AVERAGE_REWARD[-1000:]) / 1000}")
            print("# -------------------------------- #")
            # visualize_ai(test_AI)
        # visualize_ai(test_AI, diagnostics=True, render=False)

        test_AI.update_q_values(training_set)
        training_set = []

    print("Training finished!")

    # After training

    NNV.plot_smoothed(AVERAGE_REWARD, sigma=10, show=True)
    save_ai(test_AI, 'c:\\Users\\Ayyaz\\PycharmProjects\\NewNeuralNetworkProject\\stuff', 'myAI')
    visualize_ai(test_AI, terminal=False)
