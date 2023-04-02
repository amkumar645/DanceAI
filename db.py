from flask import Flask
from flask_pymongo import pymongo
import os

CONNECTION_STRING = os.getenv("db")
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.get_database('rekindl')
user_collection = pymongo.collection.Collection(db, 'users')
explore_collection = pymongo.collection.Collection(db, 'explore')
connect_collection = pymongo.collection.Collection(db, 'connect')
