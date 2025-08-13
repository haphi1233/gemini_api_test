import requests
import feedparser
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from dateutil import parser as date_parser
from urllib.parse import quote
import time

@dataclass
class NewsArticle:
    title: str
    link: str
    snippet: str
    date: str
    source: str
    position: int
    keyword: str
    language: str
    crawl_method: str

class SerperRssCrawler:
    def __init__(self, serper_api_key: str = None, limit_per_keyword: int = 10, crawl_date: datetime = None):
        """
        Initialize crawler with Serper API key
        
        Args:
            serper_api_key: API key from serper.dev
            limit_per_keyword: Maximum articles per keyword (default: 10)
            crawl_date: Reference date for last 2 days crawling (default: today)
        """
        # Serper API key (from user)
        self.serper_api_key = serper_api_key or "fcd1f8d982e3a2d4ddeb777f9e18d774d181bc14"
        
        # Crawl configuration
        self.limit_per_keyword = limit_per_keyword
        self.crawl_date = crawl_date or datetime.now()
        
        # Keywords by language
        self.korean_keywords = [
            "참치",           
            "만두",           
            "리챔",           
            "양반김",         
            "식품위생법",      
            "미세 플라스틱"    
        ]
        
        self.english_keywords = [
            "Beer Market",
            "Soju Market",
            "Korean rice wine Market",
            "Beverage Market",
            "Bottled Water Company",
            "Carbonated Beverage",
            "Sparkling Water",
            "Children Beverage",
            "Sports Drink",
            "RTD Coffee",
            "Energy Drink",
            "Health Tonic",
            "Aseptic",
            "RTD Beverage",
            "Hangover Cure"
        ]
        
        # Date configuration
        self.two_days_ago = self.crawl_date - timedelta(days=2)
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.logger = logging.getLogger(__name__)
    
    def search_with_serper(self, keyword: str, language: str = 'ko', limit: int = None) -> List[NewsArticle]:
        """
        Search news with Serper API (for Korean)
        Serper supports Korean well and returns quality results from Google
        
        Args:
            keyword: Search keyword
            language: Language ('ko' for Korean)
            limit: Maximum articles (default: self.limit_per_keyword)
        
        Returns:
            List of found articles
        """
        articles = []
        
        try:
            # Serper API endpoint
            url = f"https://google.serper.dev/news"
            
            # Parameters for Serper
            # Use provided limit or default
            article_limit = limit or self.limit_per_keyword
            
            params = {
                "q": keyword,
                "gl": "kr" if language == 'ko' else "us",  # Country
                "num": min(article_limit, 100),  # Number of results (max 100 per API)
                "tbs": "qdr:d2",  # Time filter: last 2 days
                "apiKey": self.serper_api_key
            }
            
            # Call API
            self.logger.info(f"🔍 Searching Serper for '{keyword}' (lang: {language})")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse results
                news_items = data.get('news', [])
                self.logger.info(f"   Found {len(news_items)} items from Serper")
                
                for idx, item in enumerate(news_items, 1):
                    # Stop if we reach the limit
                    if len(articles) >= article_limit:
                        break
                        
                    article = NewsArticle(
                        title=item.get('title', ''),
                        link=item.get('link', ''),
                        snippet=item.get('snippet', ''),
                        date=item.get('date', ''),
                        source=item.get('source', ''),
                        position=idx,
                        keyword=keyword,
                        language=language,
                        crawl_method='serper_api'
                    )
                    articles.append(article)
                
                self.logger.info(f"Successfully got {len(articles)} articles (limit: {article_limit})")
                
            elif response.status_code == 401:
                self.logger.error(f" Serper API key invalid or expired")
            elif response.status_code == 429:
                self.logger.error(f" Serper rate limit reached")
            else:
                self.logger.error(f"Serper API error: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error calling Serper API: {e}")
        
        return articles
    
    def search_with_google_rss(self, keyword: str, language: str = 'en', limit: int = None) -> List[NewsArticle]:
        """
        Search news with Google News RSS (for English)
        Free, no API key required
        
        Args:
            keyword: Search keyword
            language: Language ('en' for English)
            limit: Maximum articles (default: self.limit_per_keyword)
        
        Returns:
            List of found articles
        """
        articles = []
        
        try:
            # Use provided limit or default
            article_limit = limit or self.limit_per_keyword
            
            # Encode keyword for URL
            encoded_keyword = quote(keyword)
            
            # Google News RSS URL with time filter
            # when=2d means within last 2 days
            if language == 'en':
                rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:2d&hl=en-US&gl=US&ceid=US:en"
            else:
                # Fallback for other languages
                rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:2d&hl={language}"
            
            self.logger.info(f"🔍 Searching Google RSS for '{keyword}' (lang: {language}, last 2 days)")
            
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                self.logger.info(f"   Found {len(feed.entries)} items from RSS (last 2 days)")
                articles_added = 0
                
                for idx, entry in enumerate(feed.entries[:20], 1):  # Check more to get enough after filtering
                    # Parse date
                    date_str = ""
                    pub_date = None
                    
                    if hasattr(entry, 'published'):
                        try:
                            pub_date = date_parser.parse(entry.published)
                            # Normalize timezone
                            if pub_date.tzinfo:
                                pub_date = pub_date.replace(tzinfo=None)
                            
                            # Check if within 2 days from crawl_date
                            time_diff = self.crawl_date - pub_date
                            if time_diff.days > 2 or time_diff.days < 0:
                                continue  # Skip articles older than 2 days or in future
                            
                            # Keep original English date format
                            date_str = entry.published
                        except:
                            # If can't parse date, include it anyway
                            date_str = "Unknown date"
                    
                    # Extract source from title (Google News format: "Title - Source")
                    title = entry.get('title', '')
                    source = "Google News"
                    if ' - ' in title:
                        parts = title.rsplit(' - ', 1)
                        if len(parts) == 2:
                            title = parts[0]
                            source = parts[1]
                    
                    # Check basic relevance
                    if keyword.lower() in title.lower() or \
                       keyword.lower() in entry.get('summary', '').lower():
                        
                        articles_added += 1
                        article = NewsArticle(
                            title=title,
                            link=entry.get('link', ''),
                            snippet=entry.get('summary', '')[:200] if entry.get('summary') else '',
                            date=date_str,
                            source=source,
                            position=articles_added,
                            keyword=keyword,
                            language=language,
                            crawl_method='google_rss'
                        )
                        articles.append(article)
                        
                        # Limit to specified number of relevant articles
                        if articles_added >= article_limit:
                            break
                
                self.logger.info(f"  Filtered to {len(articles)} relevant articles (limit: {article_limit})")
            else:
                self.logger.info(f"   No results found")
                
        except Exception as e:
            self.logger.error(f"Error parsing Google RSS: {e}")
        
        return articles
    
    def crawl_keyword(self, keyword: str, is_korean: bool = False, limit: int = None) -> List[NewsArticle]:
        """
        Crawl news for one keyword
        
        Args:
            keyword: Keyword to search
            is_korean: True if Korean keyword
            limit: Maximum articles (default: self.limit_per_keyword)
        
        Returns:
            List of found articles
        """
        if is_korean:
            # Use Serper API for Korean (more accurate)
            return self.search_with_serper(keyword, 'ko', limit)
        else:
            # Use Google RSS for English (free)
            return self.search_with_google_rss(keyword, 'en', limit)
    
    def run(self, limit_per_keyword: int = None):
        """
        Run crawler for all keywords
        
        Args:
            limit_per_keyword: Override limit for this run
        """
        all_articles = []
        
        self.logger.info("\n" + "="*70)
        self.logger.info(" SERPER + RSS NEWS CRAWLER")
        self.logger.info("="*70)
        # Use provided limit or class default
        run_limit = limit_per_keyword or self.limit_per_keyword
        
        self.logger.info(f" Date range: Last 2 days from {self.crawl_date.strftime('%Y-%m-%d')}")
        self.logger.info(f" Articles per keyword: {run_limit}")
        self.logger.info(f"Korean keywords: Using Serper API")
        self.logger.info(f"English keywords: Using Google RSS")
        
        # Process Korean keywords with Serper
        self.logger.info(f"{'='*60}")
        self.logger.info("Processing Korean Keywords (Serper API)")
        self.logger.info(f"{'='*60}")
        
        for keyword in self.korean_keywords:
            articles = self.crawl_keyword(keyword, is_korean=True, limit=run_limit)
            all_articles.extend(articles)
            print(f"• {keyword}: {len(articles)}/{run_limit} articles")
            time.sleep(0.5)  # Rate limiting
        
        # Process English keywords with Google RSS
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Processing English Keywords (Google RSS)")
        self.logger.info(f"{'='*60}")
        
        for keyword in self.english_keywords:
            articles = self.crawl_keyword(keyword, is_korean=False, limit=run_limit)
            all_articles.extend(articles)
            print(f"• {keyword}: {len(articles)}/{run_limit} articles")
            time.sleep(0.5)  # Rate limiting
        
        # Save results
        self.save_results(all_articles)
        
        # Print summary
        self.print_summary(all_articles)
        
        return all_articles
    
    def save_results(self, articles: List[NewsArticle]):
        """
        Save results to JSON file
        """
        output = {
            'crawler': 'Serper + Google RSS Crawler',
            'crawl_time': datetime.now().isoformat(),
            'crawl_date': self.crawl_date.isoformat(),
            'total_articles': len(articles),
            'configuration': {
                'korean_keywords': 'Serper API',
                'english_keywords': 'Google RSS',
                'date_range': f'Last 2 days from {self.crawl_date.strftime("%Y-%m-%d")}',
                'limit_per_keyword': self.limit_per_keyword
            },
            'articles': []
        }
        
        # Convert articles to dict
        for article in articles:
            article_dict = asdict(article)
            output['articles'].append(article_dict)
        
        # Sort by position
        output['articles'].sort(key=lambda x: x['position'])
        
        # Save to file
        with open('serper_rss_results.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n📁 Results saved to serper_rss_results.json")
    
    def print_summary(self, articles: List[NewsArticle]):
        """
        Print summary report
        """
        self.logger.info(f"\n{'='*70}")
        self.logger.info(" CRAWLER SUMMARY")
        self.logger.info(f"{'='*70}")
        
        # Overall stats
        self.logger.info(f"\n OVERALL:")
        self.logger.info(f"  • Total articles: {len(articles)}")
        self.logger.info(f"  • Date range: Last 2 days")
        
        # By language
        ko_articles = [a for a in articles if a.language == 'ko']
        en_articles = [a for a in articles if a.language == 'en']
        
        self.logger.info(f"\n BY LANGUAGE:")
        self.logger.info(f"  • Korean (Serper): {len(ko_articles)} articles")
        self.logger.info(f"  • English (RSS): {len(en_articles)} articles")
        
        # By method
        serper_articles = [a for a in articles if a.crawl_method == 'serper_api']
        rss_articles = [a for a in articles if a.crawl_method == 'google_rss']
        
        self.logger.info(f"\n BY SOURCE:")
        self.logger.info(f"  • Serper API: {len(serper_articles)} articles")
        self.logger.info(f"  • Google RSS: {len(rss_articles)} articles")
        
        # Top keywords
        keyword_stats = {}
        for article in articles:
            if article.keyword not in keyword_stats:
                keyword_stats[article.keyword] = 0
            keyword_stats[article.keyword] += 1
        
        if keyword_stats:
            self.logger.info(f"\n TOP KEYWORDS:")
            sorted_keywords = sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            for keyword, count in sorted_keywords:
                lang_flag = "KR" if keyword in self.korean_keywords else "EN"
                self.logger.info(f"  {lang_flag} {keyword}: {count} articles")
        
        self.logger.info(f"\n{'='*70}")


def main():
    """
    Main function test crawler
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info(" Starting Serper + RSS News Crawler")
    logger.info(" Configuration:")
    logger.info("  • Korean keywords → Serper API (accurate)")
    logger.info("  • English keywords → Google RSS (free)")
    logger.info("")
    
    # Initialize crawler
    # Can pass different API key, limit, and crawl_date if needed
    # Example: crawler = SerperRssCrawler(limit_per_keyword=5, crawl_date=datetime(2025, 8, 10))
    crawler = SerperRssCrawler()
    
    # Run crawler
    # Can override limit for this run
    # Example: articles = crawler.run(limit_per_keyword=3)
    articles = crawler.run()
    
    logger.info(f"\n Crawling completed!")
    logger.info(f" Total articles: {len(articles)}")
    logger.info(f" Check 'serper_rss_results.json' for details")


if __name__ == "__main__":
    main()
