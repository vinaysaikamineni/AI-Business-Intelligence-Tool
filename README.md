# 🧠 AI-Powered SQL Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.46.0-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/openai-1.30.1-green.svg)](https://openai.com/)
[![PostgreSQL](https://img.shields.io/badge/postgresql-supported-blue.svg)](https://www.postgresql.org/)

An intelligent SQL query assistant that transforms natural language questions into SQL queries and executes them against your PostgreSQL database. Built with enterprise-grade security, comprehensive error handling, and a beautiful Streamlit interface.

![Demo Screenshot](https://via.placeholder.com/800x400/667eea/ffffff?text=AI+SQL+Assistant+Demo)

## ✨ Features

### 🔒 **Security First**
- **Query Validation**: Only allows SELECT and WITH statements
- **SQL Injection Protection**: Validates and sanitizes all queries
- **Environment-based Configuration**: Secure credential management
- **Query Timeouts**: Prevents runaway queries

### 🚀 **Advanced Functionality**
- **Natural Language Processing**: Convert English to SQL using GPT
- **Interactive Web Interface**: Beautiful, responsive Streamlit UI
- **Real-time Query Execution**: Instant results with PostgreSQL
- **Query History**: Track and replay previous queries
- **Export Options**: Download results as CSV or JSON
- **Performance Metrics**: Execution time and result statistics

### 🛠️ **Developer Experience**
- **Comprehensive Testing**: Full test suite with pytest
- **Code Quality**: Black formatting and flake8 linting
- **Error Handling**: Detailed logging and user-friendly error messages
- **Modular Architecture**: Clean, maintainable codebase

## 🏗️ Architecture

```
📁 AI_Assistant_ETL/
├── 🧠 app.py              # Main Streamlit application
├── ⚙️ config.py           # Configuration management
├── 🗄️ db_runner.py        # Database operations & security
├── 🤖 generate_sql.py     # AI-powered SQL generation
├── 🧪 test_suite.py       # Comprehensive test suite
├── 🚀 setup.py            # Automated setup script
├── 📋 requirements.txt    # Python dependencies
├── 🔐 .env.example        # Environment template
└── 📖 README.md          # This file
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
git clone https://github.com/vinaysaikamineni/ai_etl_assistant.git
cd AI_Assistant_ETL
python3 setup.py
```

### Option 2: Manual Setup

1. **Clone and Navigate**
   ```bash
   git clone https://github.com/vinaysaikamineni/ai_etl_assistant.git
   cd AI_Assistant_ETL
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chinook
DB_USER=postgres
DB_PASSWORD=your_secure_password

# Application Configuration
MAX_QUERY_HISTORY=50
QUERY_TIMEOUT=30
```

### Database Setup (Chinook)

1. **Install PostgreSQL**: [Download](https://www.postgresql.org/download/)
2. **Import Chinook Database**:
   ```bash
   psql -U postgres -d postgres < Chinook_PostgreSQL.sql
   ```
3. **Verify Tables**:
   ```sql
   \c chinook
   \dt
   ```

## 🎯 Usage

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Ask Natural Language Questions**:
   - "Show me the top 5 customers by total spending"
   - "What are the most popular music genres?"
   - "List all tracks by AC/DC"
   - "Which employee has the most customers?"
   - "Find customers who bought classical music"

3. **Review Generated SQL** - The AI will create optimized PostgreSQL queries

4. **Analyze Results** - View data in interactive tables with metrics

5. **Export Data** - Download results as CSV or JSON

## 💡 Example Queries

| Natural Language Question | Generated SQL Query |
|---------------------------|---------------------|
| "Show customers from Germany" | `select customer_id, first_name, last_name, city from customer where country = 'Germany'` |
| "Top 5 best selling artists" | `select ar.name, sum(il.unit_price * il.quantity) as total_sales from artist ar join album al on ar.artist_id = al.artist_id join track t on al.album_id = t.album_id join invoice_line il on t.track_id = il.track_id group by ar.name order by total_sales desc limit 5` |
| "Customers with jazz purchases" | `select distinct c.first_name, c.last_name, c.email from customer c join invoice i on c.customer_id = i.customer_id join invoice_line il on i.invoice_id = il.invoice_id join track t on il.track_id = t.track_id join genre g on t.genre_id = g.genre_id where g.name = 'Jazz'` |

## 🧪 Testing

```bash
# Run all tests
pytest test_suite.py -v

# Run specific test class
pytest test_suite.py::TestDatabaseSecurity -v

# Run with coverage
pytest test_suite.py --cov=. --cov-report=html
```

## 🔧 Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking (if using mypy)
mypy .
```

### Project Structure

```
AI_Assistant_ETL/
├── 📱 Frontend
│   └── app.py                 # Streamlit interface
├── 🧠 Core Logic
│   ├── config.py              # Configuration management
│   ├── generate_sql.py        # AI SQL generation
│   └── db_runner.py           # Database operations
├── 🧪 Testing
│   ├── test_suite.py          # Comprehensive tests
│   ├── app_test.py            # Legacy test file
│   ├── db_runner_test.py      # Database tests
│   └── generate_sql_test.py   # SQL generation tests
├── 🗄️ Database
│   ├── Chinook_PostgreSQL.sql # Sample database
│   └── connection_test.py     # Connection verification
├── ⚙️ Configuration
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example          # Environment template
│   ├── .gitignore            # Git exclusions
│   └── setup.py              # Automated setup
└── 📖 Documentation
    └── README.md             # This file
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production (Docker)
```dockerfile
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Connect your GitHub repo
- **Heroku**: Use the provided `Procfile`
- **AWS/GCP/Azure**: Deploy with container services

## 🛡️ Security Considerations

- ✅ **Query Validation**: Only SELECT/WITH statements allowed
- ✅ **SQL Injection Protection**: Input sanitization and validation
- ✅ **Environment Variables**: Secure credential management
- ✅ **Query Timeouts**: Prevents resource exhaustion
- ✅ **Connection Pooling**: Efficient database connections
- ✅ **Error Handling**: No sensitive information in error messages

## 📈 Performance

- **Query Caching**: Implement Redis for frequently used queries
- **Connection Pooling**: Use pgbouncer for production
- **Query Optimization**: AI learns from execution patterns
- **Result Pagination**: Handle large datasets efficiently

## 🔮 Roadmap

### Phase 1 (Current)
- [x] Basic NL to SQL conversion
- [x] PostgreSQL integration
- [x] Security framework
- [x] Comprehensive testing

### Phase 2 (Next)
- [ ] Multiple database support (MySQL, SQLite, BigQuery)
- [ ] Query caching and optimization
- [ ] Advanced data visualization
- [ ] Query explanation and optimization suggestions

### Phase 3 (Future)
- [ ] User authentication and roles
- [ ] Collaborative query sharing
- [ ] Advanced analytics and insights
- [ ] API endpoint for programmatic access

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style
- Use **Black** for formatting
- Follow **PEP 8** guidelines
- Add **type hints** where appropriate
- Write **comprehensive tests**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:

1. **Check** the [Issues](https://github.com/vinaysaikamineni/ai_etl_assistant/issues) page
2. **Search** for existing solutions
3. **Create** a new issue with detailed information
4. **Include** logs and error messages

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Streamlit** for the amazing web framework
- **PostgreSQL** for robust database support
- **Chinook Database** for sample data
- **Contributors** who make this project better

---

**Made with ❤️ by [Vinay Sai Kamineni](https://github.com/vinaysaikamineni)**
