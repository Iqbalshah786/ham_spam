from fastapi import FastAPI
from routes import email_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="German Email Spam Detector")

# Enable CORS if needed (e.g., if your frontend is running on a different domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include email routes
app.include_router(email_routes.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the German Email Spam Detector API!"}
