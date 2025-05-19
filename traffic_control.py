import pygame
import math
import numpy as np
import torch
from neural_network import TrafficControllerBrain
import time

class TrafficControl:
    def __init__(self, environment):
        self.environment = environment
        
        # Traffic signal parameters
        intersection_x, intersection_y = environment.get_nearest_intersection()
        self.intersection_pos = (intersection_x, intersection_y)
        
        # Traffic signal phases:
        # 0: North-South green, East-West red
        # 1: North-South red, East-West green
        # 2: North-South left turn green
        # 3: East-West left turn green
        self.phases = 4
        self.current_phase = 0
        
        # Traffic light positions (relative to intersection)
        road_width = environment.road_width
        self.traffic_lights = [
            # North light
            (intersection_x + road_width//3, intersection_y - road_width//2, 0),
            # South light
            (intersection_x - road_width//3, intersection_y + road_width//2, 180),
            # East light
            (intersection_x + road_width//2, intersection_y - road_width//3, 90),
            # West light
            (intersection_x - road_width//2, intersection_y + road_width//3, 270)
        ]
        
        # Neural network for traffic control
        self.brain = TrafficControllerBrain()
        
        # Timing
        self.last_phase_change = time.time()
        self.min_phase_duration = 5.0  # Minimum seconds per phase
        
        # Traffic statistics
        self.cars_per_lane = [0, 0, 0, 0]  # NSEW
        self.avg_speed_per_lane = [0, 0, 0, 0]
        self.wait_time_per_lane = [0, 0, 0, 0]
    
    def update(self):
        """Update traffic control logic."""
        # Check if it's time to potentially change the phase
        current_time = time.time()
        if current_time - self.last_phase_change >= self.min_phase_duration:
            # Prepare state for neural network decision
            state = []
            for i in range(4):
                state.append(self.cars_per_lane[i] / 10.0)  # Normalize car count
                state.append(self.avg_speed_per_lane[i])  # Already normalized
                state.append(self.wait_time_per_lane[i] / 30.0)  # Normalize wait time
            
            # Get recommendation from the neural network
            new_phase = self.brain.get_action(state)
            
            # Only change if different from current phase
            if new_phase != self.current_phase:
                self.current_phase = new_phase
                self.last_phase_change = current_time
                
                # Reset wait time for the lanes that just got a green light
                if self.current_phase == 0:  # North-South green
                    self.wait_time_per_lane[0] = 0
                    self.wait_time_per_lane[1] = 0
                elif self.current_phase == 1:  # East-West green
                    self.wait_time_per_lane[2] = 0
                    self.wait_time_per_lane[3] = 0
        
        # Increment wait time for lanes that have red lights
        if self.current_phase in [1, 2, 3]:  # North-South has red
            self.wait_time_per_lane[0] += 0.1
            self.wait_time_per_lane[1] += 0.1
        if self.current_phase in [0, 2, 3]:  # East-West has red
            self.wait_time_per_lane[2] += 0.1
            self.wait_time_per_lane[3] += 0.1
    
    def draw(self, screen):
        """Draw traffic signals."""
        for i, (x, y, angle) in enumerate(self.traffic_lights):
            # Determine light color based on current phase
            if self.current_phase == 0:  # North-South green
                color = (0, 255, 0) if i < 2 else (255, 0, 0)
            elif self.current_phase == 1:  # East-West green
                color = (0, 255, 0) if i >= 2 else (255, 0, 0)
            elif self.current_phase == 2:  # North-South left turn
                color = (255, 255, 0)  # Yellow for all in this simplified version
            else:  # East-West left turn
                color = (255, 255, 0)  # Yellow for all in this simplified version
            
            # Draw traffic light
            pygame.draw.circle(screen, color, (int(x), int(y)), 5)
    
    def should_stop(self, x, y, angle):
        """Determine if a car should stop based on traffic signals."""
        # Find the nearest intersection
        intersection_x, intersection_y = self.intersection_pos
        
        # Calculate distance to the intersection
        dx = intersection_x - x
        dy = intersection_y - y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Only consider cars close to the intersection
        if distance > 100:
            return False
        
        # Determine which lane the car is in based on its position and angle
        is_north_south = abs(math.cos(math.radians(angle))) < 0.5
        approaching_intersection = (
            (angle < 180 and dy > 0) or  # Coming from north
            (angle > 180 and dy < 0) or  # Coming from south
            (angle > 90 and angle < 270 and dx > 0) or  # Coming from west
            ((angle < 90 or angle > 270) and dx < 0)  # Coming from east
        )
        
        if not approaching_intersection:
            return False
        
        # Check if the car should stop based on the current traffic phase
        if is_north_south:
            # If in North-South lane and light is red for North-South
            return self.current_phase == 1
        else:
            # If in East-West lane and light is red for East-West
            return self.current_phase == 0
    
    def get_signal_positions(self):
        """Return the positions of all traffic signals."""
        return [(x, y) for x, y, _ in self.traffic_lights]
    
    def update_traffic_stats(self, cars):
        """Update traffic statistics based on cars in the simulation."""
        # Reset counts
        car_counts = [0, 0, 0, 0]
        total_speeds = [0, 0, 0, 0]
        
        # Process each car
        for car in cars:
            # Determine which lane the car is in based on its position and angle
            car_x, car_y = car.x, car.y
            car_angle = car.angle
            
            # Calculate distance to intersection
            intersection_x, intersection_y = self.intersection_pos
            dx = intersection_x - car_x
            dy = intersection_y - car_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Only count cars near the intersection
            if distance > 200:
                continue
            
            # Determine lane
            if abs(math.cos(math.radians(car_angle))) < 0.5:
                # North-South lane
                if car_y < intersection_y:
                    lane = 0  # North
                else:
                    lane = 1  # South
            else:
                # East-West lane
                if car_x > intersection_x:
                    lane = 2  # East
                else:
                    lane = 3  # West
            
            # Update statistics
            car_counts[lane] += 1
            total_speeds[lane] += car.speed / car.max_speed  # Normalized speed
        
        # Calculate average speeds
        for i in range(4):
            if car_counts[i] > 0:
                self.avg_speed_per_lane[i] = total_speeds[i] / car_counts[i]
            else:
                self.avg_speed_per_lane[i] = 1.0  # Default to max speed if no cars
            
            # Update car counts
            self.cars_per_lane[i] = car_counts[i]