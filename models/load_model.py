import torch
from models.rl_agent import SpamDetectorAgent  # Import the agent class

def load_model(model_path: str):
    # Instantiate the model first (adjust input_size and hidden_size as per your model)
    input_size = 512  # Example input size, adjust as needed
    hidden_size = 256  # Example hidden size, adjust as needed
    model = SpamDetectorAgent(input_size, hidden_size)
    
    # Load the state dictionary (weights)
    state_dict = torch.load(model_path, weights_only=True)  # Set weights_only=True to avoid the warning
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model
