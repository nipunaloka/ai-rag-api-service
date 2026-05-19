"""
seed_mongo.py
-------------
Run once to insert sample property data into MongoDB.
Usage: python seed_mongo.py
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)
db = client["homelanka"]
collection = db["properties"]

# Clear existing data
collection.delete_many({})

sample_properties = [
    {
        "type": "house",
        "bedrooms": 3,
        "location": "Colombo 05",
        "price": 80000,
        "description": "near school and supermarket",
        "status": "available",
    },
    {
        "type": "apartment",
        "bedrooms": 2,
        "location": "Kandy City",
        "price": 50000,
        "description": "mountain view",
        "status": "available",
    },
    {
        "type": "villa",
        "bedrooms": 4,
        "location": "Galle",
        "price": 150000,
        "description": "sea view, private pool",
        "status": "available",
    },
    {
        "type": "house",
        "bedrooms": 2,
        "location": "Negombo",
        "price": 40000,
        "description": "near beach and airport",
        "status": "available",
    },
    {
        "type": "apartment",
        "bedrooms": 1,
        "location": "Colombo 03",
        "price": 90000,
        "description": "city center location, modern",
        "status": "available",
    },
]

result = collection.insert_many(sample_properties)
print(f"Inserted {len(result.inserted_ids)} properties into MongoDB.")
client.close()