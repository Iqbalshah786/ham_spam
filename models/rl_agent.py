from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F

class SpamDetectorAgent(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is float
        x = x.float()
        return self.network(x)
    
    def act(self, state: torch.Tensor) -> int:
        with torch.inference_mode():
            # Ensure input is float and properly shaped
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension
            q_values = self(state)
            return torch.argmax(q_values, dim=1)[0].item()

class RLTrainer:
    def __init__(self, input_size: int = 768, hidden_size: int = 256):
        self.agent = SpamDetectorAgent(input_size, hidden_size)
        self.optimizer = torch.optim.AdamW(self.agent.parameters())
        self.criterion = nn.MSELoss()
    
    def train_step(self,
                  state: torch.Tensor,
                  action: int,
                  reward: float,
                  next_state: torch.Tensor) -> float:
        # Ensure inputs are float and properly shaped
        state = state.float()
        next_state = next_state.float()
        
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
            
        self.optimizer.zero_grad(set_to_none=True)
        
        # Get Q-values for current state
        current_q_values = self.agent(state)
        current_q = current_q_values[0, action]  # Select Q-value for taken action
        
        # Calculate target Q-value
        with torch.inference_mode():
            next_q_values = self.agent(next_state)
            next_q = torch.max(next_q_values[0])
        
        target_q = torch.tensor(reward + 0.99 * next_q.item(), dtype=torch.float32)
        
        # Calculate loss and update
        loss = self.criterion(current_q, target_q)
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()