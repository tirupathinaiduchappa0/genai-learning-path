import sqlite3
import json
import hashlib
import uuid
import logging

logger = logging.getLogger(__name__)

def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    logger.info("Connected to %s", db_path)
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
