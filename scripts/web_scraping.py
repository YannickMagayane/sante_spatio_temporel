import argparse
import csv
import json
import os
import sys
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Import custom geocoding
try:
    from scripts.geo_reference import geocode_location, geocode_or_default, PROVINCES
except ImportError:
    # Handle if run from different CWD
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from scripts.geo_reference import geocode_location, geocode_or_default, PROVINCES

# Default search keywords
KEYWORDS_DISEASES = ["paludisme", "choléra", "rougeole", "mpox", "ebola", "fievre jaune"]
KEYWORDS_ALERTS = ["épidémie", "éclosion", "cas suspect", "alerte sanitaire", "flambée"]

def enrich_with_geo(result, default_city="kinshasa"):
    """
    Enrich a result with lat, lon and province by analyzing text.
    """
    text_to_analyze = f"{result.get('title', '')} {result.get('text', '')}"
    geo = geocode_location(text_to_analyze)
    
    if geo:
        result.update({
            "lat": geo["lat"],
            "lon": geo["lon"],
            "province": geo["province"],
            "location_name": geo["name"]
        })
    else:
        # Fallback to current target city or default
        fallback = geocode_or_default(result.get("city", default_city))
        result.update({
            "lat": fallback["lat"],
            "lon": fallback["lon"],
            "province": fallback["province"],
            "location_name": fallback["name"]
        })
    return result

def scrape_twitter(keywords, city, max_results=50):
    results = []
    query = f"{keywords} {city} Congo"
    try:
        from ntscraper import Nitter
        scraper = Nitter()
        tweets = scraper.get_tweets(query, mode="term", number=max_results)
        if "tweets" in tweets:
            for tweet in tweets["tweets"]:
                res = {
                    "date": tweet.get("date", ""),
                    "text": tweet.get("text", ""),
                    "source": tweet.get("link", ""),
                    "city": city,
                    "platform": "twitter",
                    "keywords": keywords,
                    "scraped_at": datetime.now().isoformat(),
                }
                results.append(enrich_with_geo(res, city))
    except Exception as exc:
        print(f"❌ Twitter error: {exc}", file=sys.stderr)
    return results

def scrape_who(keywords, city):
    results = []
    base_url = "https://www.who.int/emergencies/disease-outbreak-news"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(base_url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, "lxml")
        articles = soup.find_all("a", class_="sf-list-vertical__item")
        for article in articles:
            title = article.find(["h3", "h4", "span"]).get_text(strip=True)
            link = article.get("href", "")
            if "congo" in title.lower() or "rdc" in title.lower():
                res = {
                    "date": datetime.now().strftime("%Y-%m-%d"), # Simplified
                    "title": title,
                    "text": title,
                    "source": f"https://www.who.int{link}" if not link.startswith("http") else link,
                    "city": city,
                    "platform": "oms",
                    "keywords": keywords,
                    "scraped_at": datetime.now().isoformat(),
                }
                results.append(enrich_with_geo(res, city))
    except Exception as exc:
        print(f"❌ WHO error: {exc}", file=sys.stderr)
    return results

def scrape_reliefweb(keywords, city, max_results=20):
    results = []
    api_url = "https://api.reliefweb.int/v1/reports"
    try:
        params = {
            "appname": "health-drc-analysis",
            "query[value]": f"{keywords} {city} Congo",
            "filter[field]": "country.name",
            "filter[value]": "Democratic Republic of the Congo",
            "limit": max_results,
            "fields[include][]": ["title", "date.original", "url_alias"],
        }
        resp = requests.get(api_url, params=params)
        data = resp.json()
        for item in data.get("data", []):
            f = item.get("fields", {})
            res = {
                "date": f.get("date", {}).get("original", ""),
                "title": f.get("title", ""),
                "text": f.get("title", ""),
                "source": f.get("url_alias", ""),
                "city": city,
                "platform": "reliefweb",
                "keywords": keywords,
                "scraped_at": datetime.now().isoformat(),
            }
            results.append(enrich_with_geo(res, city))
    except Exception as exc:
        print(f"❌ ReliefWeb error: {exc}", file=sys.stderr)
    return results

def save_results(results, output_path="data/scraped/"):
    if not results: return None
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"scraped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved {len(results)} results to {filepath}")
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["twitter", "oms", "reliefweb", "all"], required=True)
    parser.add_argument("--keywords", type=str, default="épidémie")
    parser.add_argument("--city", type=str, default="RDC")
    args = parser.parse_args()
    
    results = []
    if args.platform in ["twitter", "all"]: results.extend(scrape_twitter(args.keywords, args.city))
    if args.platform in ["oms", "all"]: results.extend(scrape_who(args.keywords, args.city))
    if args.platform in ["reliefweb", "all"]: results.extend(scrape_reliefweb(args.keywords, args.city))
    
    save_results(results)

