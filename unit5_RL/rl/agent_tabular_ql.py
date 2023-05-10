"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
#TRAINING_EP = .00001  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = .1  # learning rate for training
# ALPHA = .01  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    
    action_index, object_index = None, None
    
    take_rand_action = bool(np.random.binomial(1, p=epsilon))
    
    if take_rand_action:
        rng = np.random.default_rng()
        rand_action_idx = rng.choice(NUM_ACTIONS)
        rand_object_idx = rng.choice(NUM_OBJECTS)

        action_index, object_index = rand_action_idx, rand_object_idx

    else:
        current_q = q_func[state_1, state_2]
        # for max q of this state, get object and action index
        act_obj_idx_tuple = np.unravel_index(np.argmax(current_q), current_q.shape)

        action_index, object_index = act_obj_idx_tuple

    return (action_index, object_index)


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    # TODO Your code here
    # q_func[current_state_1, current_state_2, action_index,
    #        object_index] = 0  # TODO Your update here
 
    current_q = q_func[current_state_1, current_state_2, action_index,
           object_index]
    next_state_qs = q_func[next_state_1, next_state_2]
    next_state_max_q = np.amax(next_state_qs)
    if terminal:
        next_state_max_q = 0
    
    new_q = (1-ALPHA) * current_q + ALPHA * (reward + GAMMA * next_state_max_q)
    
    q_func[current_state_1, current_state_2, action_index,
           object_index] = new_q

    return None  # This function shouldn't return anything


# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0
    # initialize for each episode
    # TODO Your code here
    
    step = -1
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    while not terminal:
        step += 1
        # Choose next action and execute
        # TODO Your code here
        current_state_1 = dict_room_desc[current_room_desc]
        current_state_2 = dict_quest_desc[current_quest_desc]
        
        action_index, object_index = epsilon_greedy(current_state_1, current_state_2, q_func, epsilon)
        
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, action_index, object_index)
            
            
        if for_training:
            # update Q-function.
            # TODO Your code here
            next_state_1 = dict_room_desc[next_room_desc]
            next_state_2 = dict_quest_desc[next_quest_desc]
            
            tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                                   object_index, reward, next_state_1, next_state_2,
                                   terminal)

        if not for_training:
            # update reward
            # TODO Your code here
            epi_reward = epi_reward + GAMMA**step * reward
            
        # prepare next step
        # TODO Your code here
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc
                
    if not for_training:
        return epi_reward 


# pragma: coderesponse end


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
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

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
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
