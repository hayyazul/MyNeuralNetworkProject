import random

import numpy as np
import NeuralNetwork as NN


class QAgent:

    def __init__(self, neural_network_layers, learning_rate=0.01, alpha=0.7, discount=0.9, seed=None):
        """
        """
        self.neural_network = NN.NeuralNetwork(neural_network_layers, learning_rate, seed=seed)
        self.alpha = alpha
        self.discount = discount

    def get_q_values(self, game_array):
        """

        :param game_array: Takes in a given game array for the neural network.
        :return: The Q-Values for the given game state.
        """
        q_values = self.neural_network.predict(game_array)
        return q_values

    def update_rule(self, reward_of_state, max_q_value, action_index, q_values):
        # Returns the desired output for a q value.
        copy_of_q_values = q_values.copy()
        copy_of_q_values[action_index] = ((1 - self.alpha) * copy_of_q_values[action_index]) + \
                                         (self.alpha * (reward_of_state + self.discount * max_q_value))
        return copy_of_q_values

    def update_q_values(self, training_tuple):
        """

        :param training_tuple: An iterable containing another iterable with 2 elements: The state array and the desired
        q_value outputs.
        :return:
        """
        X = np.array([a[0] for a in training_tuple])  # This combines the training data into one array.
        Y = np.array([a[1] for a in training_tuple])  # Same as X but for the desired outputs.
        self.neural_network.epoch(X, Y)


# --- Test Classes --- #
class CountingGame:

    def __init__(self, largest_number=3):
        # The goal of this game is to count in order.
        self.current_number = 0
        self.last_number = -1
        self.largest_number = largest_number

    def make_move(self, number_to_choose):
        self.last_number = self.current_number
        self.current_number = number_to_choose


def counting_game_state_to_array(game: CountingGame):
    game_array = np.zeros(game.largest_number + 1)
    game_array[game.current_number] = 1
    return game_array


def counting_game_get_reward(game: CountingGame):
    if game.current_number == game.last_number + 1:
        return 1
    else:
        return -1


def counting_game_end(game: CountingGame):
    if game.current_number >= game.largest_number or game.current_number == game.last_number:
        return 1
    else:
        return 0


# --- Random Seeds --- #
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    test_game = CountingGame()
    agent = QAgent((NN.NeuralNetworkLayer(4, 5),
                    NN.NeuralNetworkLayer(5, 3)), alpha=1)


    for i in range(10000):
        training_data = []
        for _ in range(30):
            test_game.current_number = 0
            game_end = False
            while not game_end:
                test_game_array = counting_game_state_to_array(test_game)
                q_val = agent.get_q_values(test_game_array)
                if random.random() > 0.95:
                    move_index = random.randint(0, test_game.largest_number - 1)
                else:
                    move_index = np.argmax(q_val)

                test_game.make_move(move_index + 1)

                reward = counting_game_get_reward(test_game)
                game_end = counting_game_end(test_game)
                if game_end:
                    training_data.append((test_game_array,
                                          agent.update_rule(reward,
                                                            0,
                                                            move_index,
                                                            q_val)))
                else:
                    expected_future_reward = np.max(agent.get_q_values(counting_game_state_to_array(test_game)))
                    training_data.append((test_game_array,
                                          agent.update_rule(reward,
                                                            expected_future_reward,
                                                            move_index,
                                                            q_val)))
        agent.update_q_values(training_data)
