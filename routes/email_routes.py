from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Optional
from models.rl_agent import RLTrainer
from models.preprocessing import TextPreprocessor
from app.database.mongodb import emails_collection

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
        prediction = trainer.agent.act(state)
        
        from datetime import datetime
        emails_collection.insert_one({
            "content": email.content,
            "predicted_label": prediction,
            "actual_label": email.is_spam,
            "timestamp": datetime.utcnow()
        })
        
        return {"is_spam": prediction == 0, "confidence_score": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
