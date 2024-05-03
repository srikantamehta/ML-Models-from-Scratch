import random
import numpy as np

class Racetrack:
    """
    A class to represent a racetrack loaded from a file, managing the track layout,
    starting points, and finish lines.

    Attributes:
        track (list): A 2D list representing the racetrack where '#' is a wall,
                      '.' is an open path, 'S' is a starting line, and 'F' is a finish line.
        starting_points (list): A list of tuples (x, y) representing the coordinates
                                of all starting points on the track.
        finish_lines (list): A list of tuples (x, y) representing the coordinates
                             of all finish lines on the track.
    """
    def __init__(self, track_file):
        """
        Initializes the Racetrack object by loading the track from a file.

        Args:
            track_file (str): The file path to the racetrack configuration file.
        """
        self.track, self.starting_points, self.finish_lines = self.load_track(track_file)
    
    def load_track(self, file_path):
        """
        Loads the racetrack data from a specified file.

        Args:
            file_path (str): The path to the file containing the racetrack data.

        Returns:
            tuple: A tuple containing:
                - track (list of list of str): The racetrack grid.
                - starting_points (list of tuple): Starting points on the track.
                - finish_lines (list of tuple): Finish lines on the track.
        """    
        with open(file_path, 'r') as file:
            dimensions = file.readline().strip().split(',')
            rows, cols = int(dimensions[0]), int(dimensions[1])
            
            track = []
            starting_points = []
            finish_lines = []
            
            for y in range(rows):
                line = file.readline().strip()
                track.append(list(line))
                for x in range(cols):
                    if line[x] == 'S':
                        starting_points.append((x, y))
                    elif line[x] == 'F':
                        finish_lines.append((x, y))
                    
        return track, starting_points, finish_lines

class Car:
    """
    Represents a car on a racetrack with capabilities to move based on simulated physics and input actions.

    Attributes:
        x (int): The current x-coordinate of the car on the track.
        y (int): The current y-coordinate of the car on the track.
        vx (int): The current velocity of the car along the x-axis.
        vy (int): The current velocity of the car along the y-axis.
    """
    def __init__(self, x, y, vx=0, vy=0):
        """
        Initializes the Car with a position and initial velocity.

        Args:
            x (int): The initial x-coordinate of the car.
            y (int): The initial y-coordinate of the car.
            vx (int, optional): The initial velocity of the car along the x-axis. Defaults to 0.
            vy (int, optional): The initial velocity of the car along the y-axis. Defaults to 0.
        """
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy

    def apply_action(self, ax, ay):
        """
        Applies an acceleration action to the car, adjusting its velocity and position.
        There is a 20% chance that the acceleration will fail, meaning the velocity remains unchanged.

        Args:
            ax (int): The acceleration along the x-axis.
            ay (int): The acceleration along the y-axis.
        """
        # Implement the 20% chance of acceleration failure
        if random.random() > 0.2:  
            # Update velocity considering maximum velocity constraints
            self.vx = max(min(self.vx + ax, 5), -5)
            self.vy = max(min(self.vy + ay, 5), -5)
            
        # Update position
        self.x += self.vx
        self.y += self.vy

class Simulator:
    """
    A simulator for a racetrack environment where a car can navigate through given tracks with specific rules.

    Attributes:
        racetrack (Racetrack): The racetrack on which the car will be simulated.
        crash_option (str): Strategy to handle when the car crashes ('nearest' or any other specified strategy).
        start_x (int): Starting x-coordinate for the car on the racetrack.
        start_y (int): Starting y-coordinate for the car on the racetrack.
        car (Car): The car object being simulated.
        running (bool): Flag to indicate if the simulation is active.
        transition_table (dict): Table mapping states and actions to resulting states, rewards, and termination status.
    """
    def __init__(self, racetrack, crash_option='nearest'):
        """
        Initializes the simulator with a racetrack and crash handling option.

        Args:
            racetrack (Racetrack): The racetrack instance where the simulation will be run.
            crash_option (str, optional): Method to handle crashes. Defaults to 'nearest'.
        """
        self.racetrack = racetrack
        self.crash_option = crash_option
        self.start_x, self.start_y = random.choice(self.racetrack.starting_points)
        self.car = Car(self.start_x, self.start_y, vx=0, vy=0)
        self.running = True
        self.transition_table = self.build_transition_table()

    def build_transition_table(self):
        """
        Builds a transition table for all possible states and actions given the current racetrack configuration.

        Returns:
            dict: A dictionary representing the transition table with states, actions, and their outcomes.
        """
        transition_table = {}
        # Define possible actions in the environment
        action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        # Iterate over all possible positions and velocities within the track limits
        for x in range(len(self.racetrack.track[0])):
            for y in range(len(self.racetrack.track)):
                for vx in range(-5, 6): # velocity limits
                    for vy in range(-5, 6):
                        if self.racetrack.track[y][x] != '#': # check if not a wall
                            state = (x, y, vx, vy)
                            transition_table[state] = {}
                            for ax, ay in action_space:
                                # Compute the resulting state from applying each action
                                new_state, reward, is_terminal = self.compute_state_transition(state, (ax, ay))
                                transition_table[state][(ax, ay)] = (new_state, reward, is_terminal)
        self.reset()
        return transition_table

    def compute_state_transition(self, state, action):
        """
        Computes the next state based on the current state and action taken. Handles the dynamics of car movement,
        including possible acceleration failures and collisions.

        Args:
            state (tuple): The current state of the car, represented as (x, y, vx, vy) where x, y are the coordinates,
                        and vx, vy are the velocities in the x and y directions respectively.
            action (tuple): The action taken, represented as (ax, ay) where ax, ay are accelerations applied to the velocities.

        Returns:
            tuple: A tuple containing the new state, the reward received, and a boolean indicating if the state is terminal.
        """
        x, y, vx, vy = state
        ax, ay = action

        # Compute potential new velocity
        if random.random() > 0.2:  # 80% chance to succeed
            vx = max(min(vx + ax, 5), -5)
            vy = max(min(vy + ay, 5), -5)

        # Compute potential new position
        new_x = x + vx
        new_y = y + vy

        # Check for crashes at each intermediate position along the path
        current_x, current_y = x, y
        while current_x != new_x or current_y != new_y:
            next_x = current_x + (1 if new_x > current_x else -1 if new_x < current_x else 0)
            next_y = current_y + (1 if new_y > current_y else -1 if new_y < current_y else 0)
            if self.check_path_for_crash(current_x, current_y, next_x, next_y):
                self.handle_crash(next_x, next_y)
                new_x, new_y = self.car.x, self.car.y
                vx, vy = 0, 0  # Reset velocity
                return (new_x, new_y, vx, vy), -1, False
            current_x, current_y = next_x, next_y

        # Check if the new position is on the finish line
        if (new_x, new_y) in self.racetrack.finish_lines:
            return (new_x, new_y, 0, 0), 0, True

        return (new_x, new_y, vx, vy), -1, False


    def simulate_with_policy(self, policy, max_steps=500):
        """
        Simulates the car movement on the racetrack according to a given policy until the car reaches the finish line or
        the maximum number of steps is reached.

        Args:
            policy (dict): A mapping from states to actions.
            max_steps (int): The maximum number of steps to simulate.

        Returns:
            int: The number of steps taken to reach the goal or the maximum step limit.
        """
        self.reset() # Reset the simulation to the starting point
        state = (self.car.x, self.car.y, 0, 0)
        steps = 0
        while steps < max_steps:
            action = policy.get(state)
            if action is None:
                # No action defined for this state; end the simulation
                return steps  
            new_state, reward, is_terminal = self.compute_state_transition(state, action)
            state = new_state
            steps += 1
            if is_terminal:
                # If the state is terminal, return the number of steps taken
                return steps 
        
        # If the maximum number of steps is reached, return the step count
        return steps

    def reset(self):
        self.start_x, self.start_y = random.choice(self.racetrack.starting_points)
        self.car.x, self.car.y = self.start_x, self.start_y
        self.car.vx, self.car.vy = 0, 0  # Reset velocity to zero
        self.running = True

    def bresenhams_line_algorithm(self, x0, y0, x1, y1):
        """
        Implements Bresenham's line algorithm to compute a list of grid points between two points
        on the racetrack that the car will pass over. This method helps in checking potential crash points.

        Args:
            x0 (int): The x-coordinate of the starting point.
            y0 (int): The y-coordinate of the starting point.
            x1 (int): The x-coordinate of the ending point.
            y1 (int): The y-coordinate of the ending point.

        Returns:
            list: A list of points (tuples) representing the line from (x0, y0) to (x1, y1).
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        points.append((x1, y1))  # Add the end position to ensure the line is complete
        return points

    def check_path_for_crash(self, x0, y0, x1, y1):
        """
        Checks if the path between two points intersects with any walls ('#') on the track.

        Args:
            x0 (int): The x-coordinate of the starting point.
            y0 (int): The y-coordinate of the starting point.
            x1 (int): The x-coordinate of the ending point.
            y1 (int): The y-coordinate of the ending point.

        Returns:
            bool: True if the path intersects a wall, False otherwise.
        """
        points = self.bresenhams_line_algorithm(x0, y0, x1, y1)
        for (x, y) in points:
            if x < 0 or x >= len(self.racetrack.track[0]) or y < 0 or y >= len(self.racetrack.track) or self.racetrack.track[y][x] == '#':
                return True
        return False

    def handle_crash(self, x, y): 
        """
        Handles the event of a crash based on the specified crash_option, either resetting the car to the nearest
        valid position or to the original starting point.

        Args:
            x (int): The x-coordinate where the crash occurred.
            y (int): The y-coordinate where the crash occurred.
        """
        if self.crash_option == 'nearest':
            self.car.x, self.car.y = self.find_nearest_valid_position(x, y)
        elif self.crash_option == 'original':
            self.car.x, self.car.y = random.choice(self.racetrack.starting_points)
        # Reset velocity after handling the crash
        self.car.vx, self.car.vy = 0, 0


    def find_nearest_valid_position(self, x, y):
        """
        Finds the nearest valid position on the track that is not a wall or finish line, used for resetting the car
        post-crash.

        Args:
            x (int): The x-coordinate where the crash occurred.
            y (int): The y-coordinate where the crash occurred.

        Returns:
            tuple: The coordinates of the nearest valid position.
        """
        min_distance = float('inf')
        nearest_position = (x, y)
        for row in range(len(self.racetrack.track)):
            for col in range(len(self.racetrack.track[row])):
                if (self.racetrack.track[row][col] != '#' and  # Check for valid (non-wall) spaces
                    (row, col) not in self.racetrack.finish_lines):  # Exclude finish lines
                    distance = abs(x - col) + abs(y - row)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_position = (col, row)
        return nearest_position

class ValueIteration:
    def __init__(self, simulator, gamma=0.99, theta=0.1):
        """
        Initializes the Value Iteration algorithm with the specified simulator, discount factor,
        and convergence threshold.

        Args:
            simulator (Simulator): The simulation environment which provides the dynamics and rewards.
            gamma (float): The discount factor which balances the importance of immediate and future rewards.
            theta (float): A small threshold for determining when the value function has sufficiently converged.
        """
        self.simulator = simulator
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.states = list(simulator.transition_table.keys())
        self.V = {state: 0 for state in self.states}  # Value function initialization
        self.policy = {state: None for state in self.states}  # Initial policy
        self.iterations = 0  # Counter for iterations

    def run(self):
        """
        Executes the value iteration algorithm until the value function converges. Convergence is assumed when the 
        maximum change in value function between iterations is less than the specified threshold (theta).
        """
        while True:
            delta = 0 # Reset delta to track changes in value function
            for state in self.states:
                max_value = float('-inf') # Start with the worst possible value
                current_value = self.V[state] # Current value of the state
                # Evaluate each possible action from the current state
                for action in self.simulator.transition_table[state]:
                    total = 0
                    new_state, reward, is_terminal = self.simulator.transition_table[state][action]
                    if not is_terminal:  # If not a terminal state, proceed with the usual computation
                        total = reward + self.gamma * self.V[new_state]
                    else:
                        total = reward  # If terminal, the future value is zero
                    if total > max_value:
                        max_value = total
                        self.policy[state] = action
                # Calculate the change in value for this state
                delta = max(delta, abs(max_value - current_value)) # Update the value function for this state
                self.V[state] = max_value
            self.iterations += 1  # Increment the iteration counter
            # Check if the current changes are below the threshold for all states
            if delta < self.theta:
                break

    def get_policy(self):
        """
        Retrieves the computed optimal policy after running value iteration.

        Returns:
            dict: A dictionary mapping each state to its corresponding optimal action.
        """
        return self.policy

    def get_iterations(self):
        """
        Retrieves the number of iterations the value iteration algorithm was run until convergence.

        Returns:
            int: The number of iterations.
        """
        return self.iterations

class QLearning:
    def __init__(self, simulator, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initializes a Q-Learning agent with specific learning parameters and a simulation environment.

        Args:
            simulator (Simulator): The simulation environment which provides the dynamics and rewards.
            alpha (float): The learning rate, dictating how much the Q-values are updated during training.
            gamma (float): The discount factor, used to balance immediate and future rewards.
            epsilon (float): The initial exploration rate, determining the probability of taking a random action.
        """
        self.simulator = simulator
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.states = list(simulator.transition_table.keys()) # States in the environment
        self.actions = list(simulator.transition_table[self.states[0]].keys())  # Possible actions
        # Initialize Q-table with zero values for each state-action pair
        self.Q = {state: {action: 0 for action in self.actions} for state in self.states}
        
    def choose_action(self, state, episode):
        """
        Chooses an action using an epsilon-greedy policy. Epsilon decreases with each episode to reduce exploration over time.

        Args:
            state (tuple): The current state from which to choose an action.
            episode (int): The current episode number, used to adjust the exploration rate.

        Returns:
            action: The action chosen from the action space.
        """
        exploration_rate = self.epsilon / (episode + 1)   # Reduce exploration rate over time
        if random.random() < exploration_rate:
            return random.choice(self.actions) # Explore: choose a random action
        else:
            return max(self.Q[state], key=self.Q[state].get) # Choose the best known action
    
    def update_q(self, state, action, reward, next_state, is_terminal):
        """
        Updates the Q-value for a given state and action using the Bellman equation.

        Args:
            state (tuple): The current state.
            action: The action taken.
            reward (float): The reward received after taking the action.
            next_state (tuple): The state transitioned to after taking the action.
            is_terminal (bool): Whether the next state is a terminal state.
        """
        current_q = self.Q[state][action] # Current Q-value

        # Calculate the maximum Q-value for the next state
        if is_terminal:
            target_q = reward
        else:
            next_max_q = max(self.Q[next_state].values())  # For Q-learning
            target_q = reward + self.gamma * next_max_q  # For Q-learning
        
        # Update the Q-value using the learning rate, reward, discount factor, and maximum future Q-value
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, num_episodes):
        """
        Trains the Q-Learning agent over a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train the agent.

        Returns:
            tuple: A tuple containing the list of iterations, steps to finish, and total steps per episode.
        """
        iterations = [] # Track the number of episodes it takes to reach the goal
        steps_to_finish = [] # Track the number of steps taken to finish in each episode
        total_steps_per_episode = [] # Track total steps per episode for analysis

        for episode in range(num_episodes):
            state = (self.simulator.start_x, self.simulator.start_y, 0, 0)
            step = 0
            episode_steps = 0

            while True:
                action = self.choose_action(state, episode)  # Select an action based on the current state and episode
                next_state, reward, is_terminal = self.simulator.compute_state_transition(state, action) # Compute the state transition
                self.update_q(state, action, reward, next_state, is_terminal) # Update the Q-table based on the transition
                state = next_state  # Move to the next state
                step += 1
                episode_steps += 1

                if is_terminal:
                    iterations.append(episode + 1)  # Record the episode if the goal is reached
                    steps_to_finish.append(step) # Record the steps taken to finish
                    break

                if step >= 10000:
                    break # Prevent infinite loops by breaking after 10,000 steps

            total_steps_per_episode.append(episode_steps) # Record the total steps taken in this episode

        return iterations, steps_to_finish, total_steps_per_episode
    
    def get_policy(self):
        """
        Retrieves the learned policy as a dictionary mapping from states to actions.

        Returns:
            dict: A dictionary where keys are states and values are actions that yield the highest Q-value.
        """
        policy = {}
        for state in self.states:
            # Find the action with the maximum Q-value for each state
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy


class SARSA:
    def __init__(self, simulator, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initializes the SARSA algorithm with the given simulator settings, defining the primary attributes 
        used in the SARSA learning process.

        Args:
            simulator (Simulator): The simulation environment.
            alpha (float): Learning rate, determining how much the Q-values are updated at each step.
            gamma (float): Discount factor, quantifying the importance of future rewards.
            epsilon (float): Initial exploration rate for ε-greedy policy, guiding the trade-off between 
                             exploration and exploitation.
        """

        self.simulator = simulator
        self.alpha = alpha  # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.states = list(simulator.transition_table.keys()) # All possible states
        self.actions = list(simulator.transition_table[self.states[0]].keys()) # All possible actions
        self.Q = {state: {action: 0 for action in self.actions} for state in self.states} # Initialize Q-values

    def choose_action(self, state, episode):
        """
        Selects an action using the epsilon-greedy strategy, which balances exploration and exploitation 
        by adjusting epsilon over time.

        Args:
            state (tuple): The current state of the agent.
            episode (int): Current episode number, used to decay the exploration rate.

        Returns:
            action: The chosen action based on the current policy.
        """
        exploration_rate = self.epsilon / (episode + 1)  # Decay exploration rate over episodes
        if random.random() < exploration_rate:
            return random.choice(self.actions) # Explore: choose a random action
        else:
            return max(self.Q[state], key=self.Q[state].get) # Choose the best known action
        
    def update_q(self, state, action, reward, next_state, is_terminal,episode):
        """
        Updates the Q-value for a given state and action using the SARSA update rule.

        Args:
            state (tuple): Current state from which the action was taken.
            action (int): Action taken from the state.
            reward (float): Reward received after taking the action.
            next_state (tuple): State reached after taking the action.
            is_terminal (bool): Boolean flag indicating if the episode has ended.
            episode (int): Current episode number, used in choosing the next action.
        """
        current_q = self.Q[state][action] # Current Q-value
        if is_terminal:
            target_q = reward # Update target Q-value to reward if terminal state
        else:
            next_action = self.choose_action(next_state, episode)   # Choose next action using policy
            next_q = self.Q[next_state][next_action]  # SARSA: uses the Q-value of the chosen next action
            target_q = reward + self.gamma * next_q  # SARSA update rule
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q) # Update Q-value
    
    def train(self, num_episodes):
        """
        Trains the SARSA algorithm over a number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.

        Returns:
            tuple: A tuple containing lists of iterations, steps to finish, and total steps per episode.
        """
        iterations = [] # Track episodes completed
        steps_to_finish = [] # Track steps taken to complete each episode
        total_steps_per_episode = [] # Track all steps per episode for analysis

        for episode in range(num_episodes):
            state = (self.simulator.start_x, self.simulator.start_y, 0, 0) # Start state
            action = self.choose_action(state, episode)  # Initial action
            step = 0
            episode_steps = 0

            while True:
                next_state, reward, is_terminal = self.simulator.compute_state_transition(state, action)
                next_action = self.choose_action(next_state,episode)  # Next action from the next state
                self.update_q(state, action, reward, next_state, is_terminal,episode) # Update Q-values
                state = next_state  # Transition to next state
                action = next_action # Transition to next action
                step += 1
                episode_steps += 1

                if is_terminal:
                    iterations.append(episode + 1)
                    steps_to_finish.append(step)
                    break  # End this episode

                if step >= 10000:
                    break  # Cap on maximum steps to avoid infinite loops

            total_steps_per_episode.append(episode_steps)

        return iterations, steps_to_finish, total_steps_per_episode
    
    def get_policy(self):
        """
        Retrieves the learned policy after training. The policy dictates the best action for each state.

        Returns:
            dict: A dictionary mapping from states to actions, where each action is the one with the highest Q-value for that state.
        """
        policy = {}
        for state in self.states:
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy
