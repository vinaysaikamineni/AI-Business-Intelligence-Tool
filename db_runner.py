import psycopg2
import logging
from contextlib import contextmanager
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database errors"""
    pass

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**Config.get_db_config())
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        if conn:
            conn.rollback()
        raise DatabaseError(f"Database connection failed: {e}")
    finally:
        if conn:
            conn.close()

def validate_query(query):
    """Validate SQL query for security"""
    query_upper = query.strip().upper()
    
    # Check for allowed operations
    if not any(query_upper.startswith(op) for op in Config.ALLOWED_SQL_OPERATIONS):
        raise DatabaseError("Only SELECT and WITH queries are allowed")
    
    # Check for dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            raise DatabaseError(f"Dangerous keyword '{keyword}' not allowed")
    
    return True

def run_query(query, timeout=30):
    """Execute SQL query with security validation and timeout"""
    try:
        # Validate query
        validate_query(query)
        
        with get_db_connection() as conn:
            # Set query timeout
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{timeout}s'")
                cur.execute(query)
                
                # Get column headers
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                
                # Get rows
                rows = cur.fetchall()
                
                logger.info(f"Query executed successfully, returned {len(rows)} rows")
                return colnames, rows
                
    except DatabaseError:
        raise
    except psycopg2.Error as e:
        logger.error(f"Query execution error: {e}")
        raise DatabaseError(f"Query failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise DatabaseError(f"Unexpected error: {e}")

def test_connection():
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return result[0] == 1
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
