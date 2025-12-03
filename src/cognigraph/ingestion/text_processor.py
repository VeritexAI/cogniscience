"""
Text stream processing for diverse data sources.

Handles chunking, cleaning, and buffering of text from multiple sources
including RSS feeds, files, APIs, and web scraping.
"""

import re
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass
from datetime import datetime
import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class TextChunk:
    """Processed text chunk with metadata."""
    text: str
    source: str
    source_type: str  # 'rss', 'file', 'url', 'api', 'manual'
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextProcessor:
    """
    Process and clean text for embedding.
    
    Handles chunking, deduplication, and normalization.
    """
    
    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 500,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True
    ):
        """
        Initialize text processor.
        
        Args:
            min_length: Minimum text length to keep
            max_length: Maximum text length (will chunk longer texts)
            remove_urls: Strip URLs from text
            remove_emails: Strip email addresses
            normalize_whitespace: Normalize whitespace to single spaces
        """
        self.min_length = min_length
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emails
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
        
        # Strip and remove extra spaces
        text = text.strip()
        
        return text
    
    def chunk(self, text: str, overlap: int = 50) -> List[str]:
        """
        Split long text into chunks with overlap.
        
        Args:
            text: Text to chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_length
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end markers
                for marker in ['. ', '! ', '? ', '\n']:
                    last_marker = text.rfind(marker, start, end)
                    if last_marker > start:
                        end = last_marker + 1
                        break
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_length:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def process(self, text: str) -> List[str]:
        """
        Clean and chunk text.
        
        Args:
            text: Raw text
            
        Returns:
            List of processed text chunks
        """
        cleaned = self.clean(text)
        
        if len(cleaned) < self.min_length:
            return []
        
        return self.chunk(cleaned)


class RSSFeedReader:
    """Read and parse RSS/Atom feeds."""
    
    def __init__(self, text_processor: TextProcessor = None):
        """
        Initialize RSS feed reader.
        
        Args:
            text_processor: Optional text processor for cleaning
        """
        self.processor = text_processor or TextProcessor()
    
    def fetch(self, feed_url: str, max_entries: int = 20) -> List[TextChunk]:
        """
        Fetch and process RSS feed.
        
        Args:
            feed_url: RSS feed URL
            max_entries: Maximum entries to fetch
            
        Returns:
            List of processed text chunks
        """
        chunks = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_entries]:
                # Extract text from entry
                text_parts = []
                
                if hasattr(entry, 'title'):
                    text_parts.append(entry.title)
                
                if hasattr(entry, 'summary'):
                    # Remove HTML tags from summary
                    summary = BeautifulSoup(entry.summary, 'html.parser').get_text()
                    text_parts.append(summary)
                elif hasattr(entry, 'description'):
                    description = BeautifulSoup(entry.description, 'html.parser').get_text()
                    text_parts.append(description)
                
                combined_text = ' '.join(text_parts)
                
                # Process text
                processed = self.processor.process(combined_text)
                
                # Create chunks with metadata
                for text in processed:
                    chunk = TextChunk(
                        text=text,
                        source=feed_url,
                        source_type='rss',
                        timestamp=datetime.now(),
                        metadata={
                            'title': getattr(entry, 'title', ''),
                            'link': getattr(entry, 'link', ''),
                            'published': getattr(entry, 'published', '')
                        }
                    )
                    chunks.append(chunk)
        
        except Exception as e:
            print(f"Error fetching RSS feed {feed_url}: {e}")
        
        return chunks


class FileReader:
    """Read text from various file formats."""
    
    def __init__(self, text_processor: TextProcessor = None):
        """
        Initialize file reader.
        
        Args:
            text_processor: Optional text processor
        """
        self.processor = text_processor or TextProcessor()
    
    def read_txt(self, filepath: str) -> List[TextChunk]:
        """
        Read plain text file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            processed = self.processor.process(content)
            
            for text in processed:
                chunk = TextChunk(
                    text=text,
                    source=filepath,
                    source_type='file',
                    timestamp=datetime.now(),
                    metadata={'format': 'txt'}
                )
                chunks.append(chunk)
        
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
        
        return chunks
    
    def read_lines(self, filepath: str) -> Iterator[TextChunk]:
        """
        Read file line by line (streaming).
        
        Args:
            filepath: Path to file
            
        Yields:
            TextChunk for each line
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    processed = self.processor.process(line)
                    
                    for text in processed:
                        yield TextChunk(
                            text=text,
                            source=filepath,
                            source_type='file',
                            timestamp=datetime.now(),
                            metadata={'format': 'txt'}
                        )
        
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")


class URLReader:
    """Scrape text from web URLs."""
    
    def __init__(self, text_processor: TextProcessor = None):
        """
        Initialize URL reader.
        
        Args:
            text_processor: Optional text processor
        """
        self.processor = text_processor or TextProcessor()
    
    def fetch(self, url: str) -> List[TextChunk]:
        """
        Fetch and extract text from URL.
        
        Args:
            url: Web page URL
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'header', 'footer', 'nav']):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Process
            processed = self.processor.process(text)
            
            for text in processed:
                chunk = TextChunk(
                    text=text,
                    source=url,
                    source_type='url',
                    timestamp=datetime.now(),
                    metadata={'url': url}
                )
                chunks.append(chunk)
        
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
        
        return chunks


# Example diverse data sources
DIVERSE_RSS_FEEDS = {
    'tech': [
        'https://news.ycombinator.com/rss',
        'https://techcrunch.com/feed/',
    ],
    'news': [
        'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://feeds.bbci.co.uk/news/rss.xml',
    ],
    'science': [
        'https://www.sciencedaily.com/rss/all.xml',
        'https://www.nature.com/nature.rss',
    ],
    'business': [
        'https://www.economist.com/rss',
        'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
    ],
    'culture': [
        'https://www.theguardian.com/culture/rss',
        'https://pitchfork.com/rss/news/',
    ]
}
