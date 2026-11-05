"""Review sentiment app: Flask + Hugging Face Transformers + MongoDB."""
from __future__ import annotations
import os
from datetime import datetime, timezone
from flask import Flask, redirect, render_template_string, request, url_for
from pymongo import MongoClient, DESCENDING
from transformers import pipeline

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100_000
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'sentiment_app')
MODEL_NAME = os.getenv('MODEL_NAME', 'distilbert-base-uncased-finetuned-sst-2-english')
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
reviews = client[MONGO_DB]['reviews']
reviews.create_index([('created_at', DESCENDING)])

try:
    classifier = pipeline('sentiment-analysis', model=MODEL_NAME)
except Exception as exc:
    raise RuntimeError('Could not load the Hugging Face model. Check internet access and MODEL_NAME.') from exc

PAGE = """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Review Sentiment Analyzer</title><style>
body{font:16px system-ui;background:#f3f6fb;color:#172033;margin:0}.wrap{max-width:1050px;margin:38px auto;padding:0 18px}.card{background:white;border:1px solid #dfe6f0;border-radius:14px;padding:22px;margin:16px 0}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}textarea{box-sizing:border-box;width:100%;min-height:125px;border:1px solid #cbd5e1;border-radius:9px;padding:12px;font:inherit}button{font:inherit;background:#245de3;color:white;border:0;border-radius:8px;padding:10px 16px;cursor:pointer}.muted{color:#586579}.item{padding:12px;border-bottom:1px solid #e8edf4}.positive{color:#087443}.negative{color:#b42318}.neutral{color:#596579}.alert{background:#fff3cd;padding:12px;border-radius:8px}@media(max-width:720px){.grid{grid-template-columns:1fr}}</style>
<script src='https://cdn.jsdelivr.net/npm/chart.js'></script><div class='wrap'><h1>AI Review Sentiment Analyzer</h1><p class='muted'>Classify product reviews with a Hugging Face Transformer, save them in MongoDB, and view a sentiment report.</p>
{% if error %}<p class='alert'>{{error}}</p>{% endif %}
<section class='card'><h2>Analyze reviews</h2><form method='post' action='{{url_for("analyze")}}'><label for='reviews'>Enter one review per line (up to 20 at a time)</label><p><textarea id='reviews' name='reviews' required placeholder='The product arrived quickly and works great!'></textarea></p><button>Analyze and save</button></form><p class='muted'>The model is English-language and its scores are estimates, not guarantees.</p></section>
<div class='grid'><section class='card'><h2>Sentiment report</h2><canvas id='chart'></canvas><p><b>Total saved reviews:</b> {{total}}</p></section><section class='card'><h2>Recent results</h2>{% for r in recent %}<article class='item'><b class='{{r.sentiment|lower}}'>{{r.sentiment}}</b> · {{(r.score*100)|round|int}}%<p>{{r.text}}</p><small class='muted'>{{r.created_at.strftime('%Y-%m-%d %H:%M UTC')}}</small></article>{% else %}<p class='muted'>No reviews saved yet.</p>{% endfor %}</section></div>
<script>new Chart(document.getElementById('chart'),{type:'doughnut',data:{labels:['Positive','Negative','Neutral'],datasets:[{data:[{{counts.Positive}},{{counts.Negative}},{{counts.Neutral}}],backgroundColor:['#20a36a','#dc4c4c','#8893a5']}]},options:{plugins:{legend:{position:'bottom'}}}});</script></div></html>"""

def label_result(item: dict) -> dict:
    raw = item['label'].upper()
    sentiment = 'Neutral' if float(item['score']) < 0.60 else ('Positive' if raw in {'POSITIVE', 'LABEL_1'} else 'Negative')
    return {'sentiment': sentiment, 'score': float(item['score'])}

@app.get('/')
def index():
    error = ''
    try:
        counts = {key: reviews.count_documents({'sentiment': key}) for key in ('Positive','Negative','Neutral')}
        recent = list(reviews.find({}, {'text':1,'sentiment':1,'score':1,'created_at':1}).sort('created_at', DESCENDING).limit(25))
        total = sum(counts.values())
    except Exception:
        counts = {'Positive':0,'Negative':0,'Neutral':0}; recent=[]; total=0
        error = 'MongoDB is not reachable. Start MongoDB and refresh to view or save results.'
    return render_template_string(PAGE, counts=counts, recent=recent, total=total, error=error)

@app.post('/analyze')
def analyze():
    lines = [line.strip()[:2000] for line in request.form.get('reviews','').splitlines() if line.strip()][:20]
    if not lines: return redirect(url_for('index'))
    try:
        predictions = classifier(lines, truncation=True, max_length=512)
        documents=[]
        now=datetime.now(timezone.utc)
        for text, prediction in zip(lines, predictions):
            result=label_result(prediction)
            documents.append({'text':text, **result, 'model':MODEL_NAME, 'created_at':now})
        reviews.insert_many(documents)
        return redirect(url_for('index'))
    except Exception:
        app.logger.exception('Sentiment classification or database save failed')
        return render_template_string(PAGE, counts={'Positive':0,'Negative':0,'Neutral':0}, recent=[], total=0, error='Could not analyze or save reviews. Check that MongoDB is running and try again.'), 503

if __name__ == '__main__':
    app.run(port=int(os.getenv('PORT','5001')), debug=False)
