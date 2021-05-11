import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .variables import init_dql
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Class for a Deep Q-Learning neural network
class DQNetwork(torch.nn.Module):
    # Class constructor
    def __init__(self, random_seed, layers, net_path, train):
        super(DQNetwork, self).__init__()
        # Applies a random seed to the network
        torch.manual_seed(random_seed)

        # Stores list of layers
        self.layers = layers

        # Uses the Sequential class of Pytorch to simplify
        # the process of generating a neural net
        self.network = nn.Sequential(*layers)

        # Loads the old weights if currently testing
        if not train:
            print("Loaded saved network weights for testing...")
            self.load_state_dict(torch.load(net_path))
            self.eval()

    # Forward propogation through the neural network
    # Please not this is a required function by Pytorch
    def forward(self, x):
        return self.network(x)


# Class for the Deep Q-Learning agent
class DQLearning():
    # Class constructor
    def __init__(self, game_params):
        # Stores the parameters for the agent
        self.game_params = game_params
        self.dql_params = init_dql()

        # For saving and loading the weights of the network
        self.net_path = os.path.join('rl_algorithms/saved_agents',
                                     self.dql_params['dqnet_file'])

        # Generates and stores the state space and action space
        self.state_space = self.get_state_space()
        self.action_space = self.game_params['move_list']

        # Initializes epsilon for epsilon greedy strategy
        self.epsilon = 1

        # Creates memory for the experience replay
        self.memory = deque(maxlen=self.dql_params['memory_size'])

        # Creates the main neural networks
        self.dqnet_main = DQNetwork(self.dql_params['random_seed'],
                                    self.dql_params['layers'],
                                    self.net_path,
                                    self.game_params['train']).to(DEVICE)
        # Creates the target neurla network by making a deep copy
        self.dqnet_target = copy.deepcopy(self.dqnet_main)

        # Initializes the optimizer for the neural networks
        self.optimizer = optim.Adam(self.dqnet_main.parameters(),
                                    lr=self.dql_params['alpha'])

        # Initialize counter in order to update after a certain amount of time
        self.counter = 0

    # Uses the starting speed, max speed, and info about each type of obstacle
    # to set up the state space.
    def get_state_space(self):
        state_space = []
        start_speed = self.game_params['start_speed']
        max_speed = self.game_params['max_speed']

        # Appends an empty state for whenever no obstacles are on screen
        state_space.append((0, 0, 0, 0, 0))

        # Loops through all possible game speeds
        for s in range(start_speed, max_speed+1):
            # Loops through possible x-values for the obstacles
            for x in range(0, self.game_params['scr_width'] + 1):
                # Appends tuples for Pterodactyl obstacles
                for y in range(0, len(self.game_params['pter_y_pos'])):
                    pter_tuple = (x,
                                  self.game_params['pter_y_pos'][y],
                                  self.game_params['pter_width'],
                                  self.game_params['pter_height'],
                                  s)  # speed
                    state_space.append(pter_tuple)

                # Appends tuples for Cacti obstacles
                max_cacti_length = self.game_params['max_cacti_length']
                for n in range(1, max_cacti_length + 1):
                    # For small cacti
                    sm_cacti_tuple = (x,
                                      self.game_params['ground_pos'],
                                      n * self.game_params['sm_cacti_width'],
                                      self.game_params['sm_cacti_height'],
                                      s)
                    state_space.append(sm_cacti_tuple)

                    # For large cacti
                    lg_cacti_tuple = (x,
                                      self.game_params['ground_pos'],
                                      n * self.game_params['lg_cacti_width'],
                                      self.game_params['lg_cacti_height'],
                                      s)
                    state_space.append(lg_cacti_tuple)

        return state_space

    # Takes as input dino (a Dino class object) and obstacles (a list
    # of Obstacle class objects). If obstacles is not empty, returns a
    # tuple with the left position, bottom position, width, height, and
    # game_speed of the nearest obstacle to dino. Otherwise, returns a
    # tuple with zeros.
    def get_state(self, dino, obstacles):
        state = (0, 0, 0, 0, 0)
        # Loops through the obstacles to get the closest one
        for obs in obstacles:
            if obs.rect.right > dino.rect.left and obs.rect.left >= 0:
                state = (obs.rect.left,
                         obs.rect.bottom,
                         obs.rect.width,
                         obs.rect.height,
                         obs.speed)
                break
        return state

    # Appends the current tuple to memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Returns a random action from the action space.
    def get_random_action(self):
        return random.choice(self.action_space)

    # Takes as input a state tuple. Uses the target neural network to
    # return the action corresponding to the output node with
    # the highest value.
    def get_best_action(self, state):
        state_arr = np.asarray(state)  # Converts the state tuple to an array
        state_tensor = torch.tensor(state_arr).float().to(DEVICE)

        self.dqnet_main.eval()
        with torch.no_grad():
            prediction = self.dqnet_main(state_tensor).detach().to(DEVICE)
        self.dqnet_main.train()
        return torch.argmax(prediction).item()

    # Takes a state tuple as input. Returns an action based on the
    # epsilon greedy strategy.
    def choose_action(self, state):
        # Returns a random action if a random value is less than epsilon
        # while training.
        if random.random() < self.epsilon and self.game_params['train']:
            return self.get_random_action()
        # Otherwise, returns the best expected action.
        else:
            return self.get_best_action(state)

    # Takes as input a minibatch (a list of tuples consisting of a state
    # tuple, action integer, reward integer, next state tuple, and
    # done boolean). Loops through each element of the minibatch to
    # update the main neural network.
    def update_main(self, minibatch):
        for state, action, reward, next_state, done in minibatch:
            # Converts state tuple to array to get state tensor
            state_arr = np.asarray(state)
            state_tensor = torch.tensor(state_arr).float().to(DEVICE)

            # Converts next state tuple to array to get next state tensor
            next_state_arr = np.asarray(next_state)
            next_state_tensor = torch.tensor(next_state_arr).float().to(DEVICE)

            self.dqnet_main.train()  # Allows main neural network training
            self.dqnet_target.eval()  # Allows target neural network evaluation

            # Uses the main neural network to get the predicted output tensor
            predicted_tensor = self.dqnet_main(state_tensor)
            # Uses the target neural network to get the predicted output tensor
            # for the next state.
            with torch.no_grad():
                next_rewards = self.dqnet_target(next_state_tensor).detach()

            # Uses the given reward, gamma (the discount factor),
            # and the max value from the output of the next state
            # tensor to calculate the target value for the
            # given action.
            target = reward
            if not done:
                target += self.dql_params['gamma'] * torch.max(next_rewards).item()

            # Creates a clone of the predicted tensor.
            target_tensor = predicted_tensor.clone()
            # Replaces the value (in the cloned predicted tensor) corresponding
            # to the current action with the calculated target value.
            target_tensor[action] = target
            target_tensor.detach()  # detaches the clones target tensor

            # Uses the lost function and backward XYZ to update the main
            # neural network's weights
            criterion = torch.nn.MSELoss()
            loss = criterion(predicted_tensor, target_tensor).to(DEVICE)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # Updates the target neural networks weights by using a linear
    # interpolation between the main's weights and the target's
    # weights
    def update_target(self):
        for target_param, main_param in zip(self.dqnet_target.parameters(),
                                            self.dqnet_main.parameters()):
            weight = self.dql_params['tau'] * main_param.data + \
                      (1 - self.dql_params['tau']) * target_param.data
            target_param.data.copy_(weight)

    # Takes as input a state (tuple), action (integer),
    # dino (Dino class object), and next state (tuple).
    # Adds the given information to the experience
    # replay buffer; if the enough games have passed,
    # uses a random sample from the replay buffer
    # to train the neural networks.
    def update(self, state, action, dino, next_state):
        done = dino.has_crashed

        # Save experience in replay memory
        self.remember(state, action, dino.reward, next_state, done)

        # Learn every update_frequency time steps.
        self.counter = (self.counter + 1) % self.dql_params['update_frequency']
        if self.counter == 0:
            # If enough samples are available in memory,
            # get random subset and learn
            if len(self.memory) > self.dql_params['batch_size']:
                minibatch = random.sample(self.memory,
                                          self.dql_params['batch_size'])
                # Updates the main neural network with the random minibatch
                self.update_main(minibatch)
                # Updates the target neural network using interpolation
                self.update_target()

        # Resets the dino's rewards
        dino.reward = 0

    # Saves the weights of the target network
    def save_file(self):
        model_weights = self.dqnet_target.state_dict()
        torch.save(model_weights, self.net_path)
        print("Saved target neural network weights to path:", self.net_path)
