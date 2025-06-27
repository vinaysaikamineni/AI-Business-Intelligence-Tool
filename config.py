"""
Configuration management for AI SQL Assistant
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    
    # Database Configuration
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "chinook")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    
    # Application Configuration
    MAX_QUERY_HISTORY = int(os.getenv("MAX_QUERY_HISTORY", "50"))
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))
    
    # Security
    ALLOWED_SQL_OPERATIONS = ["SELECT", "WITH"]  # Only allow read operations
    
    @classmethod
    def get_db_config(cls):
        """Return database configuration as dictionary"""
        return {
            "host": cls.DB_HOST,
            "port": cls.DB_PORT,
            "database": cls.DB_NAME,
            "user": cls.DB_USER,
            "password": cls.DB_PASSWORD
        }
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True

# Database schema for better AI context
CHINOOK_SCHEMA = {
    "customer": ["customer_id", "first_name", "last_name", "company", "address", "city", "state", "country", "postal_code", "phone", "fax", "email", "support_rep_id"],
    "employee": ["employee_id", "last_name", "first_name", "title", "reports_to", "birth_date", "hire_date", "address", "city", "state", "country", "postal_code", "phone", "fax", "email"],
    "invoice": ["invoice_id", "customer_id", "invoice_date", "billing_address", "billing_city", "billing_state", "billing_country", "billing_postal_code", "total"],
    "invoice_line": ["invoice_line_id", "invoice_id", "track_id", "unit_price", "quantity"],
    "track": ["track_id", "name", "album_id", "media_type_id", "genre_id", "composer", "milliseconds", "bytes", "unit_price"],
    "album": ["album_id", "title", "artist_id"],
    "artist": ["artist_id", "name"],
    "genre": ["genre_id", "name"],
    "media_type": ["media_type_id", "name"],
    "playlist": ["playlist_id", "name"],
    "playlist_track": ["playlist_id", "track_id"]
}
