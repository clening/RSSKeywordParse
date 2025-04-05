#!/usr/bin/env python3
import argparse
import os
import sys
import re
import xml.etree.ElementTree as ET
import feedparser
# import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class RSSProcessor:
    def __init__(self, opml_file, output_dir, keywords=None):
        """
        Initialize the RSS processor.
        
        Args:
            opml_file (str): Path to the OPML file containing RSS feeds
            output_dir (str): Directory to store downloaded content
            keywords (list): List of keywords to search for
        """
        self.opml_file = opml_file
        self.output_dir = output_dir
        self.keywords = keywords or [keywords.txt]
        self.feeds = []
        self.results = {}

    def parse_opml(self):
        """Parse the OPML file and extract feed URLs."""
        try:
            # Read the file and fix common XML issues
            with open(self.opml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix unescaped ampersands in URLs
            fixed_content = re.sub(r'&(?!amp;|lt;|gt;|apos;|quot;)', '&amp;', content)
            
            # Parse the fixed content
            root = ET.fromstring(fixed_content)
            
            # Find all outline elements with xmlUrl attribute (RSS feeds)
            for outline in root.findall(".//outline[@xmlUrl]"):
                feed_title = outline.get("title", "Unnamed Feed")
                feed_url = outline.get("xmlUrl")
                
                if feed_url:
                    self.feeds.append({
                        "title": feed_title,
                        "url": feed_url
                    })
            
            print(f"Found {len(self.feeds)} feeds in the OPML file.")
        except Exception as e:
            print(f"Error parsing OPML file: {e}")
            sys.exit(1)

    def download_feeds(self, max_entries_per_feed=10):
        """Download and process the content of each feed."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create a directory for the raw feed content
        raw_dir = os.path.join(self.output_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        
        for feed in self.feeds:
            feed_title = feed["title"]
            feed_url = feed["url"]
            
            print(f"Processing feed: {feed_title}")
            
            try:
                # Parse the feed
                parsed_feed = feedparser.parse(feed_url)
                
                # Create a sanitized filename from the feed title
                safe_title = re.sub(r'[^\w\s-]', '', feed_title).strip().replace(' ', '_')
                
                # Save the raw feed content
                feed_dir = os.path.join(raw_dir, safe_title)
                if not os.path.exists(feed_dir):
                    os.makedirs(feed_dir)
                
                # Process each entry in the feed
                for i, entry in enumerate(parsed_feed.entries[:max_entries_per_feed]):
                    if i >= max_entries_per_feed:
                        break
                    
                    # Extract entry information
                    entry_title = entry.get("title", "Untitled")
                    entry_link = entry.get("link", "")
                    entry_date = entry.get("published", datetime.now().strftime("%Y-%m-%d"))
                    
                    # Try to get the content
                    content = ""
                    if "content" in entry:
                        content = entry.content[0].value
                    elif "summary" in entry:
                        content = entry.summary
                    elif "description" in entry:
                        content = entry.description
                    
                    # Clean the HTML content to get plain text
                    soup = BeautifulSoup(content, "html.parser")
                    clean_content = soup.get_text(separator=' ', strip=True)
                    
                    # Create a filename for this entry
                    safe_entry_title = re.sub(r'[^\w\s-]', '', entry_title).strip().replace(' ', '_')
                    entry_filename = f"{i+1:02d}_{safe_entry_title[:50]}.txt"
                    entry_path = os.path.join(feed_dir, entry_filename)
                    
                    # Save the entry content
                    with open(entry_path, "w", encoding="utf-8") as f:
                        f.write(f"Title: {entry_title}\n")
                        f.write(f"Link: {entry_link}\n")
                        f.write(f"Date: {entry_date}\n")
                        f.write(f"Feed: {feed_title}\n")
                        f.write("\n")
                        f.write(clean_content)
                    
                    # Search for keywords
                    if self.keywords:
                        for keyword in self.keywords:
                            if keyword.lower() in clean_content.lower() or keyword.lower() in entry_title.lower():
                                if keyword not in self.results:
                                    self.results[keyword] = []
                                
                                self.results[keyword].append({
                                    "feed": feed_title,
                                    "title": entry_title,
                                    "link": entry_link,
                                    "date": entry_date,
                                    "file": entry_path
                                })
            
            except Exception as e:
                print(f"Error processing feed {feed_title}: {e}")
    
    def generate_report(self):
        """Generate a report of the keywords found."""
        if not self.results:
            print("No keyword matches found.")
            return
        
        report_path = os.path.join(self.output_dir, "keyword_report.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Keyword Search Report\n")
            f.write("====================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Keywords: {', '.join(self.keywords)}\n\n")
            
            for keyword, matches in self.results.items():
                f.write(f"Keyword: {keyword}\n")
                f.write(f"Found in {len(matches)} entries\n")
                f.write("-" * 40 + "\n\n")
                
                for match in matches:
                    f.write(f"Feed: {match['feed']}\n")
                    f.write(f"Title: {match['title']}\n")
                    f.write(f"Link: {match['link']}\n")
                    f.write(f"Date: {match['date']}\n")
                    f.write(f"File: {match['file']}\n\n")
                
                f.write("\n")
        
        print(f"Keyword report generated: {report_path}")

# Load up the Keywords.txt file
def load_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Process RSS feeds from OPML and search for keywords")
    parser.add_argument("opml_file", help="Path to the OPML file containing RSS feeds")
    parser.add_argument("--output-dir", "-o", default="feed_content", help="Directory to store downloaded content")
    parser.add_argument("--file", "-f", help="User provides a file with keywords.")
    parser.add_argument("--keywords", "-k", nargs="+", help="Keywords to search for in the feed content")
    parser.add_argument("--max-entries", "-m", type=int, default=10, help="Maximum entries to process per feed")
    
    args = parser.parse_args()
    
    # Read keywords from the file
    keywords_file_path = args.file  # Adjust path if necessary
    keywords = load_keywords(keywords_file_path)
    keywords += args.keywords or []

    processor = RSSProcessor(args.opml_file, args.output_dir, keywords)
    processor.parse_opml()
    processor.download_feeds(args.max_entries)
    processor.generate_report()

if __name__ == "__main__":
    main()