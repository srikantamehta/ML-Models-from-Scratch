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


import random

class Simulator:
    def __init__(self, racetrack, crash_option='nearest'):
        self.racetrack = racetrack
        self.crash_option = crash_option
        self.start_x, self.start_y = random.choice(self.racetrack.starting_points)
        self.car = Car(self.start_x, self.start_y, vx=0, vy=0)
        self.running = True

    def reset(self):
        self.car.x, self.car.y = self.start_x, self.start_y  
        self.car.vx, self.car.vy = 0, 0
        self.running = True

    def step(self, action):
        if not self.running:
            return None, 0, True  # Return immediately if the simulator is not running.

        # Apply action with a chance of failure as defined in the Car class
        self.car.apply_action(*action)  

        # Get potential new position after applying action
        new_x, new_y = self.car.x, self.car.y

        # Check if the path to the new position results in a crash 
        if self.check_path_for_crash(self.car.x, self.car.y, new_x, new_y):
            self.handle_crash(new_x, new_y)  # Handle crash before updating position
        else:
            self.car.x, self.car.y = new_x, new_y  # Update position only if no crash

        # Define the current state after action application
        state = (self.car.x, self.car.y, self.car.vx, self.car.vy)
        next_state = state

        # Check if the car is at the finish line to determine the reward
        if (self.car.x, self.car.y) in self.racetrack.finish_lines:
            self.running = False
            reward = 0  # No penalty if finish line is reached
        else:
            reward = -1  # Standard cost for a move

        return next_state, reward, not self.running


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
                if self.racetrack.track[row][col] != '#':
                    distance = abs(x - col) + abs(y - row)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_position = (col, row)
        return nearest_position


class ValueIteration:
    def __init__(self, simulator, discount_factor=0.99, theta=0.1):
        self.simulator = simulator
        self.gamma = discount_factor
        self.theta = theta
        self.states = self.generate_states()
        self.action_space = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.values = np.zeros(len(self.states))
        self.policy = {}

    def generate_states(self):
        states = []
        for x in range(len(self.simulator.racetrack.track[0])):
            for y in range(len(self.simulator.racetrack.track)):
                for vx in range(-5, 6):
                    for vy in range(-5, 6):
                        if self.simulator.racetrack.track[y][x] != '#':
                            states.append((x, y, vx, vy))
        return states

    def run(self):
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            for state_index, state in enumerate(self.states):
                v = self.values[state_index]
                max_value = float('-inf')
                for action in self.action_space:
                    total = 0
                    transitions = self.simulator.step(state, action)
                    for prob, next_state, reward in transitions:
                        state_index_next = self.states.index(next_state)
                        total += prob * (reward + self.gamma * self.values[state_index_next])
                    if total > max_value:
                        max_value = total
                self.values[state_index] = max_value
                delta = max(delta, abs(v - self.values[state_index]))

        # Extract policy from values
        for state_index, state in enumerate(self.states):
            best_action_value = float('-inf')
            best_action = None
            for action in self.action_space:
                total = 0
                transitions = self.simulator.step(state, action)
                for prob, next_state, reward in transitions:
                    state_index_next = self.states.index(next_state)
                    total += prob * (reward + self.gamma * self.values[state_index_next])
                if total > best_action_value:
                    best_action_value = total
                    best_action = action
            self.policy[state] = best_action

    def get_policy(self):
        return self.policy

class QLearning:
    def __init__(self, simulator):
        self.simulator = simulator
        self.q_table = {}

    def train(self):
        # Implement Q-learning training logic
        pass
