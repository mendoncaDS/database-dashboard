
import os
import pytz
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine

# Constants
FREQUENCY_MAPPING = {
    "1 minute": "1T",
    "5 minutes": "5T",
    "15 minutes": "15T",
    "30 minutes": "30T",
    "1 hour": "1H",
    "2 hours": "2H",
    "4 hours": "4H",
    "6 hours": "6H",
    "12 hours": "12H",
    "1 day": "1D",
    "1 week": "1W",
    "1 month": "1M"
}

MIN_DATETIME = datetime(2017, 9, 1).replace(tzinfo=pytz.UTC)