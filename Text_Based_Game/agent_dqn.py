"""Tabular QL agent"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils as utils


DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy
    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command
    Returns:
        (int, int): the indices describing the action/object to take
    """
    if np.random.choice([1, 0], p=[epsilon, 1 - epsilon]):
        action_index = np.random.randint(0, NUM_ACTIONS)
        object_index = np.random.randint(0, NUM_OBJECTS)
    else:
        # action_index, object_index = 0, 0
        with torch.no_grad():
            q_a_curr, q_o_curr = model(state_vector)
        action_index = q_a_curr.argmax()
        object_index = q_o_curr.argmax()
    return action_index, object_index

class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


def deep_q_learning(current_state_vector, action_index, object_index, reward,
                    next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition
    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over
    Returns:
        None
    """
    with torch.no_grad():
        q_values_action_next, q_values_object_next = model(next_state_vector)

    if terminal:
        maxq_next = 0
    else:
        maxq_next = 1 / 2 * (q_values_action_next.max()
                         + q_values_object_next.max())

    q_value_cur_state = model(current_state_vector)

    q_curr = 1/2*(q_value_cur_state[0][action_index]+q_value_cur_state[1][object_index])

    loss = (reward + GAMMA*maxq_next - q_curr)**2 / 2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_episode(for_training):
    """
        Model, optimiser are freaking global
        Also dictionary. Crap.
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = 0.
    t = 0
    # initialize for each episode

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state, dictionary))
        action_index, object_index = epsilon_greedy(
            current_state_vector, epsilon
        )
        next_room_desc, next_quest_desc, reward, terminal= framework.step_game(
            current_room_desc,
            current_quest_desc,
            action_index, object_index
        )
        next_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(
                next_room_desc + next_quest_desc, dictionary
            )
        )

        if for_training:
            # update Q-function.
            deep_q_learning(
                current_state_vector,
                action_index, object_index,
                reward,
                next_state_vector,
                terminal
            )
        else:
        # if not for_training:
            # update reward
            epi_reward += (GAMMA**t)*reward

        # prepare next step
        t += 1

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)
    epoch_rewards_test.tofile("dqn_reward.dat")

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))

    axis.minorticks_on()
    axis.grid(which='major', color='k', linestyle=':')
    axis.grid(which='minor', color='k', linestyle=':')

    plt.show()
