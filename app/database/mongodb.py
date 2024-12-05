# app/database/mongodb.py  
from pymongo import MongoClient  
from app.config import settings  

client = MongoClient(settings.MONGODB_URL)  
db = client[settings.DATABASE_NAME]  
emails_collection = db["emails"]