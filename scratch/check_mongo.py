import pymongo
try:
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=2000)
    print(client.list_database_names())
except Exception as e:
    print(f"Error: {e}")
