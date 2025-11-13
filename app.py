from flask import Flask, request, render_template, flash, redirect, url_for
import os
import logging
import nltk
from textblob import TextBlob
from newspaper import Article
from datetime import datetime
from urllib.parse import urlparse
import validators
import requests

# ---------- NLTK data setup ----------
NLTK_DATA_DIR = os.getenv("NLTK_DATA_DIR", "/opt/render/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
# make sure nltk searches this directory first
nltk.data.path.insert(0, NLTK_DATA_DIR)

def ensure_nltk_models():
    """Ensure punkt and punkt_tab are available. Download if missing."""
    missing = False
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        missing = True
        nltk.download("punkt", download_dir=NLTK_DATA_DIR)
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        missing = True
        nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)
    return missing

# Try to ensure required models at startup (this may cause a short startup delay if downloading)
ensure_nltk_models()

# ---------- Flask app ----------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# secret key from env var for security; fallback to a default for local dev only
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

def get_website_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain

def safe_requests_get(url, timeout=10):
    """Requests get with a common UA to reduce being blocked, returns Response or raises."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    }
    return requests.get(url, headers=headers, timeout=timeout)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        if not validators.url(url):
            flash('Please enter a valid URL (include http/https).')
            return redirect(url_for('index'))

        # Quick HEAD/GET check to make sure the URL is reachable and not returning a 4xx/5xx
        try:
            resp = safe_requests_get(url, timeout=10)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                flash("URL does not appear to be an HTML article.")
                return redirect(url_for('index'))
        except requests.RequestException as e:
            app.logger.warning("requests failed for %s : %s", url, e)
            flash('Failed to download the content of the URL. Try another link or check the URL.')
            return redirect(url_for('index'))

        # Use newspaper to parse the article
        article = Article(url)
        try:
            article.download()
        except Exception as e:
            app.logger.error("article.download() failed: %s", e)
            flash("Failed to download article content (article.download()). Try another URL.")
            return redirect(url_for('index'))

        if not article.is_downloaded:
            app.logger.warning("article.is_downloaded is False for %s", url)
            flash("Could not retrieve the article content. Try a different URL.")
            return redirect(url_for('index'))

        try:
            article.parse()
        except Exception as e:
            app.logger.error("article.parse() failed: %s", e)
            flash("Article parsing failed. Try another URL.")
            return redirect(url_for('index'))

        # newspaper3k's nlp() sometimes fails on some sites; wrap it
        try:
            article.nlp()
        except Exception as e:
            app.logger.warning("article.nlp() failed: %s", e)
            # continue — we'll still try to extract text/title/authors

        title = article.title or "Untitled"
        authors = ', '.join(article.authors) if article.authors else get_website_name(url)
        publish_date = "N/A"
        try:
            if article.publish_date:
                publish_date = article.publish_date.strftime('%B %d, %Y')
        except Exception:
            publish_date = "N/A"

        article_text = (article.text or "").strip()
        if not article_text:
            flash("Article text is empty or could not be extracted. Try another link.")
            return redirect(url_for('index'))

        # Simple sentence-based summary fallback (newspaper's summary may not exist)
        sentences = [s.strip() for s in article_text.split('.') if s.strip()]
        max_summarized_sentences = 5
        summary = ". ".join(sentences[:max_summarized_sentences])
        if not summary:
            summary = article_text[:800] + ("..." if len(article_text) > 800 else "")

        top_image = article.top_image if getattr(article, "top_image", None) else None

        # Sentiment using TextBlob
        try:
            analysis = TextBlob(article_text)
            polarity = analysis.sentiment.polarity
        except Exception as e:
            app.logger.warning("TextBlob sentiment failed: %s", e)
            polarity = 0.0

        if polarity > 0:
            sentiment = 'happy 😊'
        elif polarity < 0:
            sentiment = 'sad 😟'
        else:
            sentiment = 'neutral 😐'

        # Pass current time for footer if template expects {{ now }}
        now = datetime.now()
        return render_template('index.html',
                               title=title,
                               authors=authors,
                               publish_date=publish_date,
                               summary=summary,
                               top_image=top_image,
                               sentiment=sentiment,
                               now=now)

    # GET
    return render_template('index.html', now=datetime.now())

if __name__ == '__main__':
    # local dev
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
