import random
import numpy as np

class Racetrack:
    def __init__(self, track_file):
        self.track, self.starting_points, self.finish_lines = self.load_track(track_file)
    
    def load_track(self, file_path):
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
    def __init__(self, x, y, vx=0, vy=0):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy

    def apply_action(self, ax, ay):
        # Implement the 20% chance of acceleration failure
        if random.random() > 0.2:  # 80% chance to succeed
            # Update velocity considering maximum velocity constraints
            self.vx = max(min(self.vx + ax, 5), -5)
            self.vy = max(min(self.vy + ay, 5), -5)
        # No else needed as velocity remains unchanged on failure
        
        # Update position
        self.x += self.vx
        self.y += self.vy

class Simulator:
    def __init__(self, racetrack, crash_option='nearest'):
        self.racetrack = racetrack
        self.crash_option = crash_option
        # Choose a random starting point from the starting points provided by the racetrack
        self.start_x, self.start_y = random.choice(self.racetrack.starting_points)
        # Initialize the Car object at the chosen starting point with initial velocity set to zero
        self.car = Car(self.start_x, self.start_y, vx=0, vy=0)
        self.running = True
        self.transition_table = self.build_transition_table()

    def build_transition_table(self):
        transition_table = {}
        action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        for x in range(len(self.racetrack.track[0])):
            for y in range(len(self.racetrack.track)):
                for vx in range(-5, 6):
                    for vy in range(-5, 6):
                        if self.racetrack.track[y][x] != '#':
                            state = (x, y, vx, vy)
                            transition_table[state] = {}
                            for ax, ay in action_space:
                                new_state, reward, is_terminal = self.compute_state_transition(state, (ax, ay))
                                # Store the new state, reward, and the is_terminal status in the transition table
                                transition_table[state][(ax, ay)] = (new_state, reward, is_terminal)
        self.reset()
        return transition_table

    def compute_state_transition(self, state, action):
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

        # Check for finish line
        if (new_x, new_y) in self.racetrack.finish_lines:
            return (new_x, new_y, 0, 0), 0, True

        return (new_x, new_y, vx, vy), -1, False


    def simulate_with_policy(self, policy, max_steps=500):
        self.reset()
        state = (self.car.x, self.car.y, 0, 0)
        steps = 0
        while steps < max_steps:
            action = policy.get(state)
            if action is None:
                # print("No action defined for this state, simulation ends.")
                return steps  # Return number of steps when no further action is possible
            new_state, reward, is_terminal = self.compute_state_transition(state, action)
            # print(f"Step {steps}: State {state} -> Action {action} -> New State {new_state}, Reward {reward}")
            state = new_state
            steps += 1
            if is_terminal:
                # print(f"Simulation ends at state {new_state} after {steps} steps.")
                return steps  # Ensure this return statement is hit when simulation ends
        
        # print(f"Simulation reaches the maximum step limit of {max_steps}.")
        return steps

    def reset(self):
        self.start_x, self.start_y = random.choice(self.racetrack.starting_points)
        self.car.x, self.car.y = self.start_x, self.start_y
        self.car.vx, self.car.vy = 0, 0  # Reset velocity to zero
        self.running = True

    def bresenhams_line_algorithm(self, x0, y0, x1, y1):
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
        points.append((x1, y1))  # Add the end position to the list of points
        return points

    def check_path_for_crash(self, x0, y0, x1, y1):
        points = self.bresenhams_line_algorithm(x0, y0, x1, y1)
        for (x, y) in points:
            if x < 0 or x >= len(self.racetrack.track[0]) or y < 0 or y >= len(self.racetrack.track) or self.racetrack.track[y][x] == '#':
                return True
        return False

    def handle_crash(self, x, y): 
        if self.crash_option == 'nearest':
            self.car.x, self.car.y = self.find_nearest_valid_position(x, y)
        elif self.crash_option == 'original':
            self.car.x, self.car.y = random.choice(self.racetrack.starting_points)
        # Reset velocity in BOTH cases
        self.car.vx, self.car.vy = 0, 0


    def find_nearest_valid_position(self, x, y):
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
        Initialize the Value Iteration algorithm with the given simulator.
        
        Args:
        - simulator (Simulator): the simulation environment
        - gamma (float): discount factor
        - theta (float): a small threshold for determining convergence
        """
        self.simulator = simulator
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Convergence threshold
        self.states = list(simulator.transition_table.keys())
        self.V = {state: 0 for state in self.states}  # Value function initialization
        self.policy = {state: None for state in self.states}  # Initial policy
        self.iterations = 0  # Counter for iterations

    def run(self):
        while True:
            delta = 0
            for state in self.states:
                max_value = float('-inf')
                current_value = self.V[state]
                for action in self.simulator.transition_table[state]:
                    total = 0
                    new_state, reward, is_terminal = self.simulator.transition_table[state][action]
                    if not is_terminal:  # If not a terminal state, proceed with the usual computation
                        total = reward + self.gamma * self.V[new_state]
                    else:
                        total = reward  # If terminal, the future value is zero (typical for terminal states)
                    if total > max_value:
                        max_value = total
                        self.policy[state] = action
                delta = max(delta, abs(max_value - current_value))
                self.V[state] = max_value
            self.iterations += 1  # Increment the iteration counter
            if delta < self.theta:
                break

    def get_policy(self):
        """
        Retrieve the computed optimal policy.

        Returns:
        - dict: a dictionary mapping from states to actions representing the optimal policy
        """
        return self.policy

    def get_iterations(self):
        """
        Retrieve the number of iterations performed during the value iteration.

        Returns:
        - int: the number of iterations
        """
        return self.iterations

class QLearning:
    def __init__(self, simulator, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the Q-Learning algorithm with the given simulator.
        """
        self.simulator = simulator
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = list(simulator.transition_table.keys())
        self.actions = list(simulator.transition_table[self.states[0]].keys())
        self.Q = {state: {action: 0 for action in self.actions} for state in self.states}
        
    def choose_action(self, state, episode):
        exploration_rate = self.epsilon / (episode + 1)  # Decay exploration rate over episodes
        if random.random() < exploration_rate:
            return random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)
    
    def update_q(self, state, action, reward, next_state, is_terminal):
        current_q = self.Q[state][action]
        if is_terminal:
            target_q = reward
        else:
            next_max_q = max(self.Q[next_state].values())  # For Q-learning
            target_q = reward + self.gamma * next_max_q  # For Q-learning
            
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, num_episodes):
        iterations = []
        steps_to_finish = []
        total_steps_per_episode = []

        for episode in range(num_episodes):
            state = (self.simulator.start_x, self.simulator.start_y, 0, 0)
            step = 0
            episode_steps = 0

            while True:
                action = self.choose_action(state, episode)  # Pass the current episode number
                next_state, reward, is_terminal = self.simulator.compute_state_transition(state, action)
                self.update_q(state, action, reward, next_state, is_terminal)
                state = next_state
                step += 1
                episode_steps += 1

                if is_terminal:
                    iterations.append(episode + 1)
                    steps_to_finish.append(step)
                    break

                if step >= 10000:
                    break

            total_steps_per_episode.append(episode_steps)

        return iterations, steps_to_finish, total_steps_per_episode
    
    def get_policy(self):
        policy = {}
        for state in self.states:
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy

    
import random

class SARSA:
    def __init__(self, simulator, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Initialize the SARSA algorithm with the given simulator.
        
        Args:
        - simulator (Simulator): the simulation environment
        - alpha (float): learning rate
        - gamma (float): discount factor
        - epsilon (float): exploration rate for ε-greedy policy
        """
        self.simulator = simulator
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Ensure all possible states and actions are initialized in the Q-table
        self.states = list(simulator.transition_table.keys())
        self.actions = list(simulator.transition_table[self.states[0]].keys())
        self.Q = {state: {action: 0 for action in self.actions} for state in self.states}

    def choose_action(self, state, episode):
        exploration_rate = self.epsilon / (episode + 1)  # Decay exploration rate over episodes
        if random.random() < exploration_rate:
            return random.choice(self.actions)
        else:
            return max(self.Q[state], key=self.Q[state].get)
        
    def update_q(self, state, action, reward, next_state, is_terminal,episode):
        current_q = self.Q[state][action]
        if is_terminal:
            target_q = reward
        else:
            next_action = self.choose_action(next_state, episode)  # Choose the next action here
            next_q = self.Q[next_state][next_action]  # For SARSA
            target_q = reward + self.gamma * next_q  # For SARSA
        self.Q[state][action] = current_q + self.alpha * (target_q - current_q)
    
    def train(self, num_episodes):
        iterations = []
        steps_to_finish = []
        total_steps_per_episode = []

        for episode in range(num_episodes):
            state = (self.simulator.start_x, self.simulator.start_y, 0, 0)
            action = self.choose_action(state, episode)  # Pass the current episode number
            step = 0
            episode_steps = 0

            while True:
                next_state, reward, is_terminal = self.simulator.compute_state_transition(state, action)
                next_action = self.choose_action(next_state,episode)
                self.update_q(state, action, reward, next_state, is_terminal,episode)
                state = next_state
                action = next_action
                step += 1
                episode_steps += 1

                if is_terminal:
                    iterations.append(episode + 1)
                    steps_to_finish.append(step)
                    break

                if step >= 10000:
                    break

            total_steps_per_episode.append(episode_steps)

        return iterations, steps_to_finish, total_steps_per_episode
    
    def get_policy(self):
        """
        Retrieve the learned policy.
        
        Returns:
        - dict: a dictionary mapping from states to actions representing the learned policy
        """
        policy = {}
        for state in self.states:
            policy[state] = max(self.Q[state], key=self.Q[state].get)
        return policy
