from pymongo import MongoClient
import gridfs
from mongoengine import connect

from model.doctor_model import Doctor

# PyMongo and GridFS
client = MongoClient("mongodb://localhost:27017/")
db = client["seranityAI"]
fs = gridfs.GridFS(db)

# MongoEngine connection (used by models)
connect(db="seranityAI", host="localhost", port=27017, alias="default")

Doctor.ensure_indexes()
