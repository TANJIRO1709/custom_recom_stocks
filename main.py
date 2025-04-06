from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from yahooquery import Ticker
import numpy as np

app = FastAPI()

# Stock Descriptions
stocks = {
    "AAPL": "Apple is a technology company that designs and sells smartphones, laptops, and software.",
    "MSFT": "Microsoft develops software, services, devices, and solutions worldwide.",
    "GOOGL": "Google is known for its search engine, cloud computing, and online advertising technologies.",
    "AMZN": "Amazon is an e-commerce and cloud computing company with a strong logistics network.",
    "TSLA": "Tesla designs, develops, and manufactures electric vehicles and energy storage products.",
    "META": "Meta Platforms develops social media technologies and virtual reality platforms.",
    "NFLX": "Netflix is a streaming service that offers a wide variety of award-winning TV shows, movies, anime, documentaries, and more.",
    "DIS": "Disney is a diversified international family entertainment and media enterprise.",
    "TCS.NS": "Tata Consultancy Services is an IT services, consulting, and business solutions organization.",
    "INFY.NS": "Infosys is a global leader in next-generation digital services and consulting.",
    "HDFCBANK.NS": "HDFC Bank is a leading private sector bank in India offering a wide range of financial services.",
    "ICICIBANK.NS": "ICICI Bank is a leading private sector bank in India providing a wide range of banking products and financial services.",
    "HINDUNILVR.NS": "Hindustan Unilever is a consumer goods company with a wide range of products in India.",
    "LT.NS": "Larsen & Toubro is a major technology, engineering, construction, manufacturing, and financial services conglomerate.",
    "ITC.NS": "ITC Limited is a diversified conglomerate with a presence in FMCG, hotels, packaging, paperboards, and agribusiness.",
}
tickers = list(stocks.keys())
descriptions = list(stocks.values())

# Realtime Data Fetch
def get_yquery(tickers):
    ticker = Ticker(tickers)
    summary = ticker.summary_detail
    details = {}
    for symbol in summary:
        try:
            beta = summary[symbol].get('beta', 0) or 0
            mcap = summary[symbol].get('marketCap', 0) or 0
            payout = summary[symbol].get('payoutRatio', 0) or 0
            details[symbol] = [beta, mcap, payout]
        except:
            details[symbol] = [0, 0, 0]
    return details

@app.get("/")
def home():
    return {"message": "Welcome to the Stock Recommender API!"}

@app.get("/recommend/{stock}")
def recommend(stock: str, top_n: int = 5, alpha: float = 0.7):
    if stock not in stocks:
        raise HTTPException(status_code=404, detail="Stock not found.")

    realtime_data = get_yquery(tickers)

    # Step 1: TF-IDF similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(descriptions)
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Step 2: Feature-based similarity
    features = [realtime_data.get(t, [0, 0, 0]) for t in tickers]
    scaler = MinMaxScaler()
    norm_features = scaler.fit_transform(features)
    feature_sim = cosine_similarity(norm_features)

    # Step 3: Combined similarity
    combined_sim = alpha * content_sim + (1 - alpha) * feature_sim
    stock_idx = tickers.index(stock)
    sim_scores = list(enumerate(combined_sim[stock_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores:
        if i != stock_idx and len(recommendations) < top_n:
            recommendations.append({
                "symbol": tickers[i],
                "score": round(float(score), 3),
                "description": stocks[tickers[i]]
            })

    return {"recommendations": recommendations}
