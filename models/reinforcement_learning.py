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
        self.lookup_table = {}
        self.populate_lookup_table()
        self.running = True

    def reset(self):
        self.car.x, self.car.y = self.start_x, self.start_y
        self.car.vx, self.car.vy = 0, 0
        self.running = True

    def populate_lookup_table(self):
        for y in range(len(self.racetrack.track)):
            for x in range(len(self.racetrack.track[0])):
                for vx in range(-5, 6):
                    for vy in range(-5, 6):
                        for ax in [-1, 0, 1]:
                            for ay in [-1, 0, 1]:
                                self.process_state_action(x, y, vx, vy, ax, ay)
        Simulator.reset(self)

    def process_state_action(self, x, y, vx, vy, ax, ay):
        # Create a temporary car to simulate the state transition
        temp_car = Car(x, y, vx, vy)
        temp_car.apply_action(ax, ay)  # Apply action with potential failure

        new_vx, new_vy = temp_car.vx, temp_car.vy
        new_x, new_y = temp_car.x, temp_car.y

        # Check if there was an actual attempt to move
        was_moving = (ax != 0 or ay != 0)

        if self.check_path_for_crash(x, y, new_x, new_y):
            # Pass the was_moving flag to handle the crash accordingly
            new_x, new_y = self.handle_crash(new_x, new_y, was_moving)
            reward = -1
        elif (new_x, new_y) in self.racetrack.finish_lines:
            reward = 0
        else:
            reward = -1

        self.lookup_table[(x, y, vx, vy, ax, ay)] = ((new_x, new_y, new_vx, new_vy), reward)


    def step(self, action):
        if not self.running:
            return None, 0, True  # Return immediately if the simulator is not running.

        # Apply action with a chance of failure as defined in the Car class
        self.car.apply_action(*action)

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

    def handle_crash(self, x, y, was_moving):
        if was_moving:
            if self.crash_option == 'nearest':
                nearest_valid = self.find_nearest_valid_position(x, y)
                self.car.x, self.car.y = nearest_valid
            elif self.crash_option == 'original':
                self.car.x, self.car.y = self.start_x, self.start_y
        # Ensure to reset velocity to 0 after a crash
        self.car.vx, self.car.vy = 0, 0
        return (self.car.x, self.car.y)


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
    def __init__(self, simulator):
        self.simulator = simulator
        self.values = np.zeros((len(simulator.track.track), len(simulator.track.track[0])))

    def run(self):
        # Implement the value iteration update logic
        pass

class QLearning:
    def __init__(self, simulator):
        self.simulator = simulator
        self.q_table = {}

    def train(self):
        # Implement Q-learning training logic
        pass
