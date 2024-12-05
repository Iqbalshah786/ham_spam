# test_db.py  
from app.database.mongodb import client  

try:  
    # Check connection  
    client.admin.command('ping')  
    print("MongoDB connection successful!")  
    
    # List databases  
    print("\nAvailable databases:")  
    print(client.list_database_names())  
except Exception as e:  
    print(f"MongoDB connection failed: {e}")