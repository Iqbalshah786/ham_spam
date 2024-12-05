import pandas as pd
import torch
from models.rl_agent import RLTrainer
from models.preprocessing import TextPreprocessor
from pathlib import Path
from transformers import BertTokenizer

def train_model(data_path: str = "data/emails.csv", epochs: int = 5):
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor and tokenizer
    preprocessor = TextPreprocessor()
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    
    # Get input size from a sample
    sample_state = preprocessor.preprocess(df['email'].iloc[0])
    input_size = len(sample_state['input_ids'].flatten())
    
    # Initialize trainer with correct input size
    trainer = RLTrainer(input_size=input_size)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for idx, row in df.iterrows():
            email_content = row['email']
            label = int(row['label'])  # 0 for spam, 1 for ham
            
            # Preprocess
            state = preprocessor.preprocess(email_content)
            
            # Ensure state is a tensor and has correct shape
            if not isinstance(state['input_ids'], torch.Tensor):
                state['input_ids'] = torch.tensor(state['input_ids'])
            
            # Flatten and validate input
            input_tensor = state['input_ids'].flatten()
            if len(input_tensor) != trainer.agent.network[0].in_features:
                raise ValueError(
                    f"Input tensor size {len(input_tensor)} doesn't match "
                    f"expected size {trainer.agent.network[0].in_features}"
                )
            
            # Get model prediction
            action = trainer.agent.act(input_tensor)  # act() should return predicted class index
            
            # Calculate reward
            reward = 1.0 if action == label else -1.0
            
            # Train step
            loss = trainer.train_step(
                input_tensor,  # Use processed tensor
                action,
                reward,
                input_tensor  # For simplicity, using same state as next_state
            )
            total_loss += loss
            
            # Print progress every 100 samples
            if (idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Sample {idx+1}/{len(df)}, "
                      f"Loss: {loss:.4f}")
        
        # Print epoch summary
        avg_loss = total_loss / len(df)
        print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path("models/saved_model")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.agent.state_dict(), save_path / "spam_detector.pth")
    print(f"Model saved to {save_path / 'spam_detector.pth'}")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
