# AI Based Review Sentiment Analyzer

A Flask web app that uses a Hugging Face Transformer to classify product reviews, stores each review and result in MongoDB, and shows a chart of positive, negative, and neutral results.

## Requirements

- Python 3.10+
- MongoDB Community running locally, or a MongoDB Atlas connection string
- Internet access for the first Transformer model download (about 250 MB)

## Windows setup

1. Install and start MongoDB, or create a free MongoDB Atlas database.
2. Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MONGO_URI="mongodb://localhost:27017"
$env:MONGO_DB="sentiment_app"
python app.py
```

For Atlas, replace the `MONGO_URI` value with your connection string. Do not commit a real connection string or password to GitHub. The model is downloaded automatically the first time the app starts. Open http://127.0.0.1:5001.

macOS/Linux: activate with `source .venv/bin/activate` and set variables with `export MONGO_URI=...` and `export MONGO_DB=sentiment_app`.

## Features

- Submit up to 20 English reviews at a time, one per line.
- Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` model returns a positive or negative class and confidence score. Low-confidence results (below 60%) are shown as neutral.
- Reviews, labels, model name, score, and timestamp are saved in MongoDB.
- Dashboard chart summarizes saved positive, negative, and neutral reviews; recent results are listed.

## Limitations

The selected pretrained model was trained for binary English sentiment. Neutral is inferred from low confidence rather than learned as a separate class. Sarcasm, other languages, and context may lead to errors. Do not use sentiment as the sole basis for important decisions. The chart loads Chart.js from a CDN, so an internet connection is needed to display it.

## Files

- `app.py` — Flask interface, Transformer prediction, and MongoDB operations
- `.env.example` — configuration example (do not add secrets to GitHub)
- `requirements.txt` — Python dependencies
