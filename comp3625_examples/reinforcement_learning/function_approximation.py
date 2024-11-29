from environment import TwoChoiceTaskWithContext
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import random
import numpy as np
import matplotlib.pyplot as plt

# In this lab you'll implement a very basic form of function approximation
# You agent is in a room with a big red light on the wall, and two buttons it can press. At each step it must observe
# whether the light is on or off (i.e. state = 1 or 0) and decide which button to press (button #0 or button #1).
# The agent must learn that when the light is "on", button 0 has a reward probability of 75% compared to 25% for
# button 1. When the light is "off", those probabilities are reversed.

# instantiate the environment object. Get the starting state
env = TwoChoiceTaskWithContext()
state, _ = env.reset()

# NOTE:
# the standard RL formulation of utility (as in the slides) is:  Q(s, a) = E[r + 𝛾 * Q(s')]
# however,
# this is NOT actually a sequential-decision task. Each decision is a one-off that doesn't affect the next state.
# So we can reduce the formulation to simply, Q(s, a) = E[r]
# that is, the value of executing action a in state s is simply the expected/average value of the reward for that action
# (this is not the case for your assignment. There you'll have to use the standard formulation)

# Following the general function approximation framework in the slides, you will:
# - set up "history" or "memory" data structures to record the agent's experiences. one for each action.
# - set up supervised learning models to predict Q values (or expected reward, in this case) given a state.
#   (one supervised model per action)
# - run the general algorithm from the slides


# instantiate the "history" data structures - one for each action. One simple option is to use dictionaries of lists:
histories = [
    {'state': [], 'reward': []},
    {'state': [], 'reward': []},
]
# so, if the agent is in state 0, executes action 1, and gets reward of 0.5, we'd add that to the history for action 1:
#   histories[1]['state'].append(0)
#   histories[1]['reward'].append(0.5)

# instantiate the two supervised learning models.
# I suggest using either LinearRegression or KNeighborsRegressor for this lab
models = [
    LinearRegression(),  # this model predicts Q value for action 0 given the state
    LinearRegression(),  # this model predicts Q value for action 1 given the state
]

# now run the general algorithm as in the slides (but with our simpler formulation of value)
rewards_gained = []
for step in range(250):

    # select an action by:
    # 1) using the models to generate Q values for each action given the current state
    # 2) choose an action based on the Q values (using e-greedy selection, boltzmann selection, etc)
    # NOTE that during the first few steps the models will be untrained. You have to keep track of when they get trained
    # You won't be able to use them for action selection until they're fit. In the meantime just choose a random action

    # YOUR CODE HERE

    # execute the chosen action in the environment
    new_state, reward, _, _, _ = env.step(selected_action)
    rewards_gained.append(reward)

    # add the experience to the history
    histories[selected_action]['state'].append(state)
    histories[selected_action]['reward'].append(reward)

    # periodically fit the models to better predict Q (or, in this case, r) from s
    if step > 0 and step % 20 == 0:
        for action_index in [0, 1]:
            x =   # needs to be in numpy array format
            y =   # needs to be in numpy array format
            models[action_index].fit(x, y)

    state = new_state

# print out the model's predicted values for each action in each state
# what *should* they be?
print('\t\t\t\taction 0\taction 1')
for state in [0, 1]:
    state = np.array(state).reshape((-1, 1))
    act0_value = models[0].predict(state)
    act1_value = models[1].predict(state)
    print(f'state {state}:\t{act0_value}\t{act1_value}')

# print out the cumulative reward curve
plt.plot(np.array(rewards_gained).cumsum())
plt.xlabel('step')
plt.ylabel('cumulative reward')
plt.show()
