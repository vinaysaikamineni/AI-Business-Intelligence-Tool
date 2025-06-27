"""
Comprehensive test suite for AI SQL Assistant
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from config import Config, CHINOOK_SCHEMA
from db_runner import run_query, test_connection, validate_query, DatabaseError
from generate_sql import prompt_to_sql, SQLGenerationError

class TestConfig:
    """Test configuration management"""
    
    def test_config_validation_success(self):
        """Test successful config validation"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            assert Config.validate_config() == True
    
    def test_config_validation_failure(self):
        """Test config validation failure"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                Config.validate_config()
    
    def test_db_config_format(self):
        """Test database configuration format"""
        db_config = Config.get_db_config()
        required_keys = ['host', 'port', 'database', 'user', 'password']
        assert all(key in db_config for key in required_keys)

class TestDatabaseSecurity:
    """Test database security features"""
    
    def test_validate_select_query(self):
        """Test SELECT query validation"""
        assert validate_query("SELECT * FROM customer") == True
    
    def test_validate_with_query(self):
        """Test WITH query validation"""
        assert validate_query("WITH cte AS (SELECT * FROM customer) SELECT * FROM cte") == True
    
    def test_reject_dangerous_queries(self):
        """Test rejection of dangerous SQL operations"""
        dangerous_queries = [
            "DROP TABLE customer",
            "DELETE FROM customer",
            "INSERT INTO customer VALUES (1, 'test')",
            "UPDATE customer SET name = 'test'",
            "ALTER TABLE customer ADD COLUMN test VARCHAR(50)",
            "CREATE TABLE test (id INT)",
            "TRUNCATE TABLE customer"
        ]
        
        for query in dangerous_queries:
            with pytest.raises(DatabaseError):
                validate_query(query)
    
    def test_reject_non_select_queries(self):
        """Test rejection of non-SELECT/WITH queries"""
        with pytest.raises(DatabaseError):
            validate_query("EXPLAIN SELECT * FROM customer")

class TestSQLGeneration:
    """Test SQL generation functionality"""
    
    @patch('generate_sql.OpenAI')
    def test_prompt_to_sql_success(self, mock_openai):
        """Test successful SQL generation"""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SELECT * FROM customer"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            result = prompt_to_sql("Show me all customers")
            assert result == "SELECT * FROM customer"
    
    @patch('generate_sql.OpenAI')
    def test_prompt_to_sql_cleanup(self, mock_openai):
        """Test SQL cleanup functionality"""
        # Mock OpenAI response with formatting
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```sql\\nSELECT * FROM customer\\n```"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            result = prompt_to_sql("Show me all customers")
            assert result == "SELECT * FROM customer"
    
    def test_sql_generation_error(self):
        """Test SQL generation error handling"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SQLGenerationError):
                prompt_to_sql("Show me all customers")

class TestDatabaseConnection:
    """Test database connection functionality"""
    
    @patch('db_runner.psycopg2.connect')
    def test_connection_success(self, mock_connect):
        """Test successful database connection"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        assert test_connection() == True
    
    @patch('db_runner.psycopg2.connect')
    def test_connection_failure(self, mock_connect):
        """Test database connection failure"""
        mock_connect.side_effect = Exception("Connection failed")
        assert test_connection() == False

class TestQueryExecution:
    """Test query execution functionality"""
    
    @patch('db_runner.get_db_connection')
    def test_successful_query_execution(self, mock_get_db_connection):
        """Test successful query execution"""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [('id',), ('name',)]
        mock_cursor.fetchall.return_value = [(1, 'John'), (2, 'Jane')]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_conn
        
        headers, rows = run_query("SELECT id, name FROM customer LIMIT 2")
        
        assert headers == ['id', 'name']
        assert rows == [(1, 'John'), (2, 'Jane')]
    
    def test_query_validation_in_execution(self):
        """Test that query validation is enforced during execution"""
        with pytest.raises(DatabaseError):
            run_query("DROP TABLE customer")

class TestSchema:
    """Test schema definitions"""
    
    def test_schema_completeness(self):
        """Test that all required tables are in schema"""
        required_tables = [
            'customer', 'employee', 'invoice', 'invoice_line',
            'track', 'album', 'artist', 'genre', 'media_type',
            'playlist', 'playlist_track'
        ]
        
        for table in required_tables:
            assert table in CHINOOK_SCHEMA
            assert isinstance(CHINOOK_SCHEMA[table], list)
            assert len(CHINOOK_SCHEMA[table]) > 0

# Integration tests
class TestIntegration:
    """Integration tests for the entire system"""
    
    @patch('db_runner.get_db_connection')
    @patch('generate_sql.OpenAI')
    def test_end_to_end_query_flow(self, mock_openai, mock_get_db_connection):
        """Test complete query flow from NL to results"""
        # Mock SQL generation
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "SELECT first_name, last_name FROM customer LIMIT 5"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        # Mock database execution
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.description = [('first_name',), ('last_name',)]
        mock_cursor.fetchall.return_value = [('John', 'Doe'), ('Jane', 'Smith')]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_get_db_connection.return_value.__enter__.return_value = mock_conn
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Generate SQL
            sql = prompt_to_sql("Show me customer names")
            
            # Execute query
            headers, rows = run_query(sql)
            
            assert headers == ['first_name', 'last_name']
            assert len(rows) == 2

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
