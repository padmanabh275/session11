import wikipediaapi
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def collect_wikipedia_articles():
    wiki = wikipediaapi.Wikipedia(
        'Hindi BPE Tokenizer/1.0 (padmanabhbosamia@gmail.com) python-wikipediaapi',
        'hi'
    )
    
    # Get popular Hindi articles - limit to main content only
    categories = ['भारत', 'विज्ञान', 'इतिहास', 'साहित्य']
    texts = []
    
    for category in categories:
        page = wiki.page(category)
        texts.append(page.text)
        
        # Limit to first 5 links per category
        for i, link in enumerate(list(page.links.items())[:5]):
            link_page = wiki.page(link[0])
            if link_page.exists():
                texts.append(link_page.text)
    
    return '\n'.join(texts)

def collect_news_articles():
    # Focus on one reliable news source
    url = 'https://www.bhaskar.com/'
    
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return '\n'.join([p.text for p in paragraphs])
    except:
        return ""

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Collect and save training data
    wiki_text = collect_wikipedia_articles()
    news_text = collect_news_articles()
    
    with open("data/hindi_corpus.txt", "w", encoding="utf-8") as f:
        f.write(wiki_text + "\n" + news_text)
    
    # Save a portion for testing
    with open("data/hindi_test.txt", "w", encoding="utf-8") as f:
        f.write(wiki_text[:1000]) 