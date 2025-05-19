import pygame
import math
import random

class Environment:
    def __init__(self, width, height):
        self.WIDTH = width
        self.HEIGHT = height
        
        # Define road layout
        self.init_layout()
        
        # Colors
        self.ROAD_COLOR = (80, 80, 80)
        self.ROAD_MARKING_COLOR = (255, 255, 255)
        self.GRASS_COLOR = (100, 220, 100)
    
    def init_layout(self):
        """Initialize the road layout."""
        # Road parameters
        self.road_width = 60
        
        # Create a crossroad in the center
        self.horizontal_road = pygame.Rect(
            0, self.HEIGHT // 2 - self.road_width // 2, 
            self.WIDTH, self.road_width
        )
        
        self.vertical_road = pygame.Rect(
            self.WIDTH // 2 - self.road_width // 2, 0, 
            self.road_width, self.HEIGHT
        )
        
        # Intersection area
        self.intersection = pygame.Rect(
            self.horizontal_road.left, self.vertical_road.top,
            self.vertical_road.width, self.horizontal_road.height
        )
        
        # Define walls (areas cars can't drive on)
        self.walls = []
        
        # Entry and exit points
        self.entry_points = [
            # (x, y, angle) for each entry point
            (10, self.HEIGHT // 2 - self.road_width // 4, 0),  # Left entry
            (self.WIDTH - 10, self.HEIGHT // 2 + self.road_width // 4, 180),  # Right entry
            (self.WIDTH // 2 - self.road_width // 4, 10, 90),  # Top entry
            (self.WIDTH // 2 + self.road_width // 4, self.HEIGHT - 10, 270)  # Bottom entry
        ]
        
        self.exit_points = [
            # (x, y) for each exit point
            (self.WIDTH - 10, self.HEIGHT // 2 - self.road_width // 4),  # Right exit
            (10, self.HEIGHT // 2 + self.road_width // 4),  # Left exit
            (self.WIDTH // 2 + self.road_width // 4, 10),  # Top exit
            (self.WIDTH // 2 - self.road_width // 4, self.HEIGHT - 10)  # Bottom exit
        ]
    
    def draw(self, screen):
        """Draw the environment."""
        # Draw grass background
        screen.fill(self.GRASS_COLOR)
        
        # Draw roads
        pygame.draw.rect(screen, self.ROAD_COLOR, self.horizontal_road)
        pygame.draw.rect(screen, self.ROAD_COLOR, self.vertical_road)
        
        # Draw road markings
        self._draw_road_markings(screen)
        
        # Draw entry and exit points
        self._draw_entry_exit_points(screen)
    
    def _draw_road_markings(self, screen):
        """Draw road markings on the roads."""
        # Horizontal road markings
        marking_width = 3
        marking_length = 20
        gap_length = 15
        total_length = marking_length + gap_length
        
        # Center line for horizontal road
        y = self.HEIGHT // 2
        for x in range(0, self.WIDTH, total_length):
            if not self.intersection.collidepoint(x + marking_length // 2, y):
                pygame.draw.line(
                    screen, self.ROAD_MARKING_COLOR, 
                    (x, y), (x + marking_length, y), 
                    marking_width
                )
        
        # Center line for vertical road
        x = self.WIDTH // 2
        for y in range(0, self.HEIGHT, total_length):
            if not self.intersection.collidepoint(x, y + marking_length // 2):
                pygame.draw.line(
                    screen, self.ROAD_MARKING_COLOR, 
                    (x, y), (x, y + marking_length), 
                    marking_width
                )
    
    def _draw_entry_exit_points(self, screen):
        """Draw entry and exit points."""
        for point in self.entry_points:
            pygame.draw.circle(screen, (0, 200, 0), (int(point[0]), int(point[1])), 5)
        
        for point in self.exit_points:
            pygame.draw.circle(screen, (200, 0, 0), (int(point[0]), int(point[1])), 5)
    
    def get_entry_points(self):
        """Return the entry points."""
        return self.entry_points
    
    def get_exit_points(self):
        """Return the exit points."""
        return self.exit_points
    
    def get_wall_distance(self, x, y, angle, max_distance):
        """Calculate distance to the nearest wall in a given direction."""
        # Convert angle to radians
        angle_rad = math.radians(angle)
        
        # Step size for ray casting
        step_size = 2
        
        # Cast a ray in the given direction
        for i in range(0, int(max_distance), step_size):
            check_x = x + i * math.cos(angle_rad)
            check_y = y + i * math.sin(angle_rad)
            
            # Check if point is on a road
            if not self._is_on_road(check_x, check_y):
                return i
        
        return max_distance
    
    def _is_on_road(self, x, y):
        """Check if a point is on the road."""
        return (self.horizontal_road.collidepoint(x, y) or 
                self.vertical_road.collidepoint(x, y))
    
    def is_colliding_with_wall(self, x, y, size):
        """Check if a car is colliding with any walls."""
        # Simple check if the car's center is on a road
        return not self._is_on_road(x, y)
    
    def get_nearest_intersection(self):
        """Get the intersection coordinates."""
        # In this simple environment, there's only one intersection
        intersection_x = self.WIDTH // 2
        intersection_y = self.HEIGHT // 2
        
        return (intersection_x, intersection_y)