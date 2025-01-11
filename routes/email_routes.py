#routes/email_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional
from models.rl_agent import RLTrainer
from models.preprocessing import TextPreprocessor
from app.database.mongodb import emails_collection
import torch
from datetime import datetime

# Initialize the router
router = APIRouter()

# Initialize preprocessor and RL trainer
preprocessor = TextPreprocessor()
trainer = RLTrainer()

# Define the Email model
class Email(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    content: str
    is_spam: Optional[bool] = None

@router.post("/predict")
async def predict_spam(email: Email):
    try:
        # Get embeddings from the text
        embeddings = preprocessor.get_embeddings(email.content)
        
        # Pass embeddings to the agent
        state = embeddings.flatten()
        q_values = trainer.agent(state)  # Get Q-values from the agent
        
        # Apply softmax to get probabilities and get confidence score
        probabilities = torch.softmax(q_values, dim=-1)
        confidence_score = probabilities.max().item()  # Max probability as confidence
        prediction = probabilities.argmax().item()  # Predicted class index (0 for spam, 1 for ham)
        
        # Insert prediction details into the MongoDB collection
        emails_collection.insert_one({
            "content": email.content,
            "predicted_label": prediction,
            "actual_label": email.is_spam,
            "timestamp": datetime.utcnow(),
            "confidence_score": confidence_score
        })
        
        # Return the response with confidence score
        return {"is_spam": prediction == 0, "confidence_score": confidence_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
