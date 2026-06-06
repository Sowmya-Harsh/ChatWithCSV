# 📊 Chat with your CSV

Upload any CSV or Excel file and ask questions in plain English. Get instant answers and interactive charts — no coding required.

## Live Demo
👉 [Try it on Streamlit Cloud](#) *(add your link after deployment)*

## Features
- Upload CSV or Excel files
- Ask questions in natural language
- Auto-generates Plotly charts (bar, line, pie, scatter, histogram)
- Understands follow-up questions with chat history
- Data quality checks (missing values, column types, stats)
- Works with any dataset — sales, finance, health, HR, and more

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq (LLaMA 3.3 70B) — free and fast
- **Charts:** Plotly Express
- **Data:** Pandas

## Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Sowmya-Harsh/chat-with-csv.git
cd chat-with-csv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your Groq API key
Get a free API key at https://console.groq.com

Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` under **Settings > Secrets**
5. Deploy!

## Example Questions
- "What are the column names and data types?"
- "Show me a bar chart of sales by region"
- "What is the average revenue per category?"
- "Are there any missing values?"
- "Plot the distribution of prices"
- "Which month had the highest total sales?"

## Project Structure
```
chat-with-csv/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```

## Author
**Sowmya Janmahanthi**
- GitHub: [Sowmya-Harsh](https://github.com/Sowmya-Harsh)
- LinkedIn: [sowmyajanmahanthi](https://linkedin.com/in/sowmyajanmahanthi)
- Hugging Face: [sowmya4547](https://huggingface.co/sowmya4547)
