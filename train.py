#train.py
import pandas as pd
import torch
from pathlib import Path
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from models.rl_agent import RLTrainer
from models.preprocessing import TextPreprocessor

def train_model(data_path: str = "data/emails.csv", epochs: int = 5, batch_size: int = 32):
    # Load data
    df = pd.read_csv(data_path)
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} samples, Validating on {len(val_data)} samples.")
    
    # Initialize preprocessor and tokenizer
    preprocessor = TextPreprocessor()
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    
    # Get input size from a sample
    sample_state = preprocessor.preprocess(train_data['email'].iloc[0])
    input_size = len(sample_state['input_ids'].flatten())
    print(f"Input size detected: {input_size}")
    
    # Initialize RLTrainer
    trainer = RLTrainer(input_size=input_size)
    
    for epoch in range(epochs):
        total_loss = 0
        for idx, row in train_data.iterrows():
            try:
                email_content = row['email']
                label = int(row['label'])  # 0 for spam, 1 for ham
                
                # Preprocess
                state = preprocessor.preprocess(email_content)
                input_tensor = torch.tensor(state['input_ids']).flatten()
                
                # Validate input size
                if len(input_tensor) != trainer.agent.network[0].in_features:
                    raise ValueError(
                        f"Input tensor size {len(input_tensor)} doesn't match expected size {trainer.agent.network[0].in_features}"
                    )
                
                # Model prediction and reward
                action = trainer.agent.act(input_tensor)
                reward = 1.0 if action == label else -1.0
                
                # Train step
                loss = trainer.train_step(input_tensor, action, reward, input_tensor)
                total_loss += loss
                
                if (idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}, Sample {idx+1}/{len(train_data)}, Loss: {loss:.4f}")
            except Exception as err:
                print(f"Error processing sample {idx+1}: {err}")
        
        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
    
    # Save model
    save_path = Path("models/saved_model")
    save_path.mkdir(parents=True, exist_ok=True)
    model_file = save_path / "spam_detector.pth"
    torch.save(trainer.agent.state_dict(), model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
