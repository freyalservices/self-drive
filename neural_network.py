import torch
import torch.nn as nn
import torch.nn.functional as F

class CarBrain(nn.Module):
    """
    Neural network that serves as the brain for each self-driving car.
    Inputs:
    - Distance to obstacles in 5 directions (front, front-left, front-right, left, right)
    - Distance to nearest traffic signal
    - Current speed
    - Angle to destination
    
    Outputs:
    - Acceleration value (-1 to 1)
    - Steering angle (-1 to 1)
    """
    
    def __init__(self):
        super(CarBrain, self).__init__()
        # Input features:
        # - 5 distance sensors
        # - Distance to traffic signal
        # - Current speed
        # - Angle to destination
        input_size = 8
        
        # Hidden layers
        hidden_size = 16
        
        # Output: acceleration and steering
        output_size = 2
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Pass input through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Use tanh to constrain outputs to (-1, 1)
        return x
    
    def get_action(self, state):
        """
        Convert state to tensor and get an action.
        Returns acceleration and steering values.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action from network
        with torch.no_grad():  # No need to track gradients during inference
            action = self.forward(state_tensor)
        
        # Extract acceleration and steering
        acceleration = action[0].item()
        steering = action[1].item()
        
        return acceleration, steering

class TrafficControllerBrain(nn.Module):
    """
    Neural network for the traffic controller to optimize traffic flow.
    Inputs:
    - Number of cars in each lane
    - Average speed of cars in each lane
    - Wait time at each intersection
    
    Outputs:
    - Traffic signal phase selection probabilities
    """
    
    def __init__(self, num_roads=4):
        super(TrafficControllerBrain, self).__init__()
        # Input features per road (cars count, avg speed, wait time)
        features_per_road = 3
        input_size = num_roads * features_per_road
        
        # Hidden layers
        hidden_size = 32
        
        # Output: probability for each traffic phase
        # For a 4-way intersection, we have 4 main phases
        output_size = 4
        
        # Network architecture
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        # Pass input through network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)  # Use softmax to get probabilities
        return x
    
    def get_action(self, state):
        """
        Convert state to tensor and get an action.
        Returns the recommended traffic phase.
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action from network
        with torch.no_grad():  # No need to track gradients during inference
            phase_probs = self.forward(state_tensor)
        
        # Choose action probabilistically
        phase = torch.multinomial(phase_probs, 1).item()
        
        return phase