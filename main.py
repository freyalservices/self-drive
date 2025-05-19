import pygame
import sys
import random
import torch
import numpy as np
from car import Car
from environment import Environment
from traffic_control import TrafficControl

# Initialize PyGame
pygame.init()
WIDTH, HEIGHT = 1000, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Self-Driving Car Simulation with AI Traffic Control")
clock = pygame.time.Clock()

# Seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Create environment and traffic control
environment = Environment(WIDTH, HEIGHT)
traffic_controller = TrafficControl(environment)

# Create initial cars
cars = []
for i in range(5):  # Start with 5 cars
    car = Car(environment, traffic_controller, random_init=True)
    cars.append(car)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font for displaying information
font = pygame.font.SysFont('Arial', 16)

# Main simulation loop
running = True
frame_count = 0
spawn_interval = 300  # Spawn new car every 300 frames

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Spawn a new car on space key
                car = Car(environment, traffic_controller, random_init=True)
                cars.append(car)
            elif event.key == pygame.K_r:  # Reset simulation
                cars = []
                for i in range(5):
                    car = Car(environment, traffic_controller, random_init=True)
                    cars.append(car)
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw environment
    environment.draw(screen)
    
    # Periodically spawn new cars if total is less than 15
    frame_count += 1
    if frame_count % spawn_interval == 0 and len(cars) < 15:
        car = Car(environment, traffic_controller, random_init=True)
        cars.append(car)
    
    # Update and draw traffic controller elements
    traffic_controller.update()
    traffic_controller.draw(screen)
    
    # Update and draw cars
    cars_to_remove = []
    for i, car in enumerate(cars):
        car.update(cars)
        car.draw(screen)
        
        # Remove cars that have reached their destination
        if car.reached_destination:
            cars_to_remove.append(i)
    
    # Remove cars that have reached their destination
    for i in sorted(cars_to_remove, reverse=True):
        cars.pop(i)
    
    # Display stats
    stats_text = [
        f"Cars: {len(cars)}",
        f"FPS: {int(clock.get_fps())}",
        f"Traffic Signal State: {traffic_controller.current_phase}",
    ]
    
    for i, text in enumerate(stats_text):
        text_surface = font.render(text, True, BLACK)
        screen.blit(text_surface, (10, 10 + i * 20))
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

# Clean up
pygame.quit()
sys.exit()