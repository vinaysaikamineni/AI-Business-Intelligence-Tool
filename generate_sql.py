import logging
from openai import OpenAI
from config import Config, CHINOOK_SCHEMA

logger = logging.getLogger(__name__)

class SQLGenerationError(Exception):
    """Custom exception for SQL generation errors"""
    pass

def create_schema_context():
    """Create detailed schema context for AI"""
    schema_text = "Database Schema (Chinook):\n\n"
    for table, columns in CHINOOK_SCHEMA.items():
        schema_text += f"{table}: {', '.join(columns)}\n"
    
    return schema_text

def create_system_prompt():
    """Create comprehensive system prompt for AI"""
    return f"""
You are an expert PostgreSQL developer working with the Chinook music database.

{create_schema_context()}

IMPORTANT RULES:
1. Only generate SELECT and WITH statements
2. Use exact table and column names from the schema above
3. Use proper PostgreSQL syntax
4. Include appropriate JOINs when querying multiple tables
5. Use LIMIT for potentially large result sets
6. Return only the SQL query, no explanations or formatting
7. Use lowercase for SQL keywords (select, from, where, etc.)
8. Always use proper table aliases for complex queries

CRITICAL DISTINCTIONS:
- artist.name = Band/Artist name (e.g., "AC/DC", "Queen", "Beatles")
- track.composer = Individual songwriter names (e.g., "Angus Young, Malcolm Young")
- When users ask for tracks "by [artist]", use artist.name, NOT track.composer
- When users ask for tracks "composed by [person]", use track.composer

Common relationships:
- customer.customer_id → invoice.customer_id
- invoice.invoice_id → invoice_line.invoice_id
- track.track_id → invoice_line.track_id
- album.album_id → track.album_id
- artist.artist_id → album.artist_id
- genre.genre_id → track.genre_id
- employee.employee_id → customer.support_rep_id
"""

def prompt_to_sql(user_prompt):
    """Convert natural language to SQL with enhanced prompting"""
    try:
        Config.validate_config()
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": create_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=Config.OPENAI_TEMPERATURE,
            max_tokens=500
        )
        
        sql = response.choices[0].message.content.strip()
        
        # Clean up common formatting issues
        sql = sql.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"Generated SQL for prompt: {user_prompt[:50]}...")
        return sql
        
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        raise SQLGenerationError(f"Failed to generate SQL: {e}")
