import os
import sys
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_user(user_id):
    logger.info("Fetching user %s", user_id)
    return {"id": user_id, "name": "Tirupathi", "created": str(datetime.now())}

def list_users():
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
