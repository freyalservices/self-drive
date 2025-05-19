import pygame
import math
import numpy as np
import torch
import random
from neural_network import CarBrain

class Car:
    def __init__(self, environment, traffic_controller, random_init=False):
        # Car dimensions
        self.width = 10
        self.length = 20
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        # Environment reference
        self.environment = environment
        self.traffic_controller = traffic_controller
        
        # Initialize position and orientation
        if random_init:
            # Position on one of the entry points
            entry_points = environment.get_entry_points()
            start_point = random.choice(entry_points)
            self.x, self.y = start_point[:2]
            self.angle = start_point[2]  # Angle in degrees
            
            # Choose a random exit point as destination
            exit_points = environment.get_exit_points()
            self.destination = random.choice(exit_points)
        else:
            # Default starting position
            self.x = 50
            self.y = environment.HEIGHT // 2
            self.angle = 0  # Angle in degrees
            self.destination = (environment.WIDTH - 50, environment.HEIGHT // 2)
        
        # Movement parameters
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0
        self.steering = 0
        self.sensor_range = 150
        
        # Neural network brain
        self.brain = CarBrain()
        
        # Status flags
        self.reached_destination = False
        self.crashed = False
        
        # Visualization properties
        self.show_sensors = True
        self.sensor_lines = []
    
    def update(self, other_cars):
        if self.reached_destination or self.crashed:
            return
        
        # Get sensor inputs
        sensor_data = self._get_sensor_data(other_cars)
        
        # Prepare state for neural network
        angle_to_destination = self._get_angle_to_destination()
        distance_to_signal = self._get_distance_to_nearest_signal()
        
        state = [
            *sensor_data,  # 5 sensor values
            distance_to_signal,
            self.speed / self.max_speed,  # Normalize speed
            angle_to_destination / 180.0,  # Normalize angle
        ]
        
        # Get action from neural network
        acceleration, steering = self.brain.get_action(state)
        
        # Apply actions
        self.acceleration = acceleration
        self.steering = steering * 5  # Scale steering for better control
        
        # Update speed based on acceleration
        self.speed += self.acceleration * 0.1
        self.speed = max(-self.max_speed/2, min(self.max_speed, self.speed))
        
        # Update angle based on steering
        self.angle += self.steering * (self.speed / self.max_speed)
        
        # Update position
        self.x += self.speed * math.cos(math.radians(self.angle))
        self.y += self.speed * math.sin(math.radians(self.angle))
        
        # Check for collisions
        if self._check_collision(other_cars):
            self.crashed = True
            self.speed = 0
            return
        
        # Check if reached destination
        if self._check_destination():
            self.reached_destination = True
            self.speed = 0
        
        # Obey traffic signals
        if self._should_stop_for_signal():
            self.speed = max(0, self.speed - 0.5)  # Gradual slowdown
    
    def draw(self, screen):
        # Draw the car as a rotated rectangle
        car_rect = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        car_rect.fill(self.color)
        car_rect = pygame.transform.rotate(car_rect, -self.angle)
        rect = car_rect.get_rect(center=(self.x, self.y))
        screen.blit(car_rect, rect)
        
        # Draw sensors if enabled
        if self.show_sensors:
            for line in self.sensor_lines:
                pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), line, 1)
        
        # Draw car direction indicator (small line pointing forward)
        indicator_end = (
            self.x + 15 * math.cos(math.radians(self.angle)),
            self.y + 15 * math.sin(math.radians(self.angle))
        )
        pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), indicator_end, 2)
    
    def _get_sensor_data(self, other_cars):
        """Get distances from sensors in 5 directions."""
        sensor_angles = [-90, -45, 0, 45, 90]  # Relative to car's heading
        sensor_data = []
        self.sensor_lines = []
        
        for angle_offset in sensor_angles:
            # Calculate absolute angle for this sensor
            sensor_angle = self.angle + angle_offset
            
            # Calculate endpoint for max sensor range
            end_x = self.x + self.sensor_range * math.cos(math.radians(sensor_angle))
            end_y = self.y + self.sensor_range * math.sin(math.radians(sensor_angle))
            
            # Default to max range
            min_distance = self.sensor_range
            
            # Check for intersection with walls
            wall_distance = self.environment.get_wall_distance(self.x, self.y, sensor_angle, self.sensor_range)
            min_distance = min(min_distance, wall_distance)
            
            # Check for intersection with other cars
            for other_car in other_cars:
                if other_car is not self:
                    car_distance = self._get_distance_to_car(sensor_angle, other_car)
                    min_distance = min(min_distance, car_distance)
            
            # Save sensor reading and visualization
            sensor_data.append(min_distance / self.sensor_range)  # Normalize to 0-1
            
            # Update endpoint for visualization based on actual distance
            end_x = self.x + min_distance * math.cos(math.radians(sensor_angle))
            end_y = self.y + min_distance * math.sin(math.radians(sensor_angle))
            self.sensor_lines.append((end_x, end_y))
        
        return sensor_data
    
    def _get_distance_to_car(self, sensor_angle, other_car):
        """Calculate distance to another car in the direction of the sensor."""
        dx = other_car.x - self.x
        dy = other_car.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate angle to the other car
        angle_to_car = math.degrees(math.atan2(dy, dx))
        
        # Check if the car is in sensor's direction (within a tolerance)
        angle_diff = abs((angle_to_car - sensor_angle + 180) % 360 - 180)
        if angle_diff <= 15 and distance <= self.sensor_range:
            return distance
        return self.sensor_range
    
    def _check_collision(self, other_cars):
        """Check for collisions with walls and other cars."""
        # Check collision with walls
        if self.environment.is_colliding_with_wall(self.x, self.y, self.width):
            return True
        
        # Check collision with other cars
        for other_car in other_cars:
            if other_car is not self:
                dx = other_car.x - self.x
                dy = other_car.y - self.y
                distance = math.sqrt(dx**2 + dy**2)
                
                # Simple collision detection based on distance
                if distance < (self.width + other_car.width) / 2:
                    return True
        
        return False
    
    def _check_destination(self):
        """Check if car has reached its destination."""
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance < 30  # Destination radius
    
    def _get_angle_to_destination(self):
        """Calculate angle to destination relative to current heading."""
        dx = self.destination[0] - self.x
        dy = self.destination[1] - self.y
        angle_to_dest = math.degrees(math.atan2(dy, dx))
        
        # Calculate relative angle
        rel_angle = (angle_to_dest - self.angle + 180) % 360 - 180
        return rel_angle
    
    def _get_distance_to_nearest_signal(self):
        """Calculate distance to the nearest traffic signal."""
        signals = self.traffic_controller.get_signal_positions()
        
        if not signals:
            return self.sensor_range  # No signals
        
        min_distance = float('inf')
        for signal_pos in signals:
            dx = signal_pos[0] - self.x
            dy = signal_pos[1] - self.y
            distance = math.sqrt(dx**2 + dy**2)
            min_distance = min(min_distance, distance)
        
        return min(min_distance, self.sensor_range) / self.sensor_range  # Normalize
    
    def _should_stop_for_signal(self):
        """Determine if the car should stop for a red light."""
        return self.traffic_controller.should_stop(self.x, self.y, self.angle)