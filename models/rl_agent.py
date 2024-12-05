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
        x = x.float()  # Ensure input is float
        return self.network(x)
    
    def act(self, state: torch.Tensor) -> int:
        """Select an action based on the current state."""
        self.eval()  # Ensure evaluation mode for inference
        with torch.inference_mode():
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension if necessary
            q_values = self(state)
            return torch.argmax(q_values, dim=1)[0].item()


class RLTrainer:
    def __init__(self, input_size: int = 768, hidden_size: int = 256, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = SpamDetectorAgent(input_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def train_step(self, 
                   state: torch.Tensor, 
                   action: int, 
                   reward: float, 
                   next_state: torch.Tensor) -> float:
        """
        Perform a single training step for the agent.
        Args:
            state (torch.Tensor): Current state input.
            action (int): Action taken by the agent.
            reward (float): Reward received for the action.
            next_state (torch.Tensor): Next state after the action.
        Returns:
            float: Loss value for the current training step.
        """
        # Move data to device
        state, next_state = state.to(self.device), next_state.to(self.device)
        
        # Ensure input tensors are in proper shape
        state = state.unsqueeze(0) if state.dim() == 1 else state
        next_state = next_state.unsqueeze(0) if next_state.dim() == 1 else next_state
        
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass for current state
        current_q_values = self.agent(state)
        current_q = current_q_values[0, action]  # Q-value for the taken action

        # Calculate target Q-value
        with torch.inference_mode():
            next_q_values = self.agent(next_state)
            next_q = torch.max(next_q_values[0])
        
        # Reward clipping for stability
        target_q = torch.tensor(reward + 0.99 * next_q.item(), dtype=torch.float32).clamp(-1.0, 1.0).to(self.device)
        
        # Loss and backpropagation
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()

        return loss.item()
