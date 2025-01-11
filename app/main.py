#/app/main.py

from fastapi import FastAPI, HTTPException
from routes import email_routes
from fastapi.middleware.cors import CORSMiddleware
from models.preprocessing import TextPreprocessor
from models.rl_agent import RLTrainer
from models.load_model import load_model
import logging


# Initialize FastAPI app
app = FastAPI(title="German Email Spam Detector")

# Enable CORS with specific allowed origins (for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include email routes
app.include_router(email_routes.router, prefix="/api/v1")

# Basic route to check if the API is working
@app.get("/")
def read_root():
    return {"message": "Welcome to the German Email Spam Detector API!"}

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Perform a simple health check
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Set up logging for better monitoring
logging.basicConfig(level=logging.INFO)

# Example of model loading at startup (if model is large)
@app.on_event("startup")
async def startup_event():
    logging.info("Starting the application...")
    # Add model loading logic here (e.g., load RLTrainer, BERT, etc.)
    model = load_model("models/saved_model/spam_detector.pth")


    preprocessor = TextPreprocessor()
    trainer = RLTrainer()
    logging.info("Model loaded and ready!")

# Example of shutdown logic
@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Shutting down the application...")
    # Cleanup resources here if needed (e.g., close database connections)

