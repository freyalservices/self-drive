import math
import numpy as np

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def angle_between_points(p1, p2):
    """Calculate angle in degrees between two points."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def normalize_angle(angle):
    """Normalize angle to be between -180 and 180 degrees."""
    return (angle + 180) % 360 - 180

def point_in_rect(point, rect):
    """Check if a point is inside a rectangle."""
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def rotate_point(point, pivot, angle):
    """Rotate a point around a pivot point by angle in degrees."""
    # Convert to radians
    angle_rad = math.radians(angle)
    
    # Translate point to origin
    px, py = point
    ox, oy = pivot
    qx = px - ox
    qy = py - oy
    
    # Rotate point
    rx = qx * math.cos(angle_rad) - qy * math.sin(angle_rad)
    ry = qx * math.sin(angle_rad) + qy * math.cos(angle_rad)
    
    # Translate point back
    return (rx + ox, ry + oy)

def clamp(value, min_value, max_value):
    """Clamp a value between minimum and maximum values."""
    return max(min_value, min(value, max_value))