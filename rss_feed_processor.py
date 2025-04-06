#!/usr/bin/env python3
import argparse
import os
import sys
import re
import xml.etree.ElementTree as ET
import feedparser
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
        self.keywords = keywords or []
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

    def get_context(self, text, keyword, context_length=50):
        """
        Get the context around a keyword in text.
        
        Args:
            text (str): The text to search in
            keyword (str): The keyword to find
            context_length (int): Number of characters before and after the keyword
            
        Returns:
            str: Context snippet with the keyword highlighted
        """
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        matches = []
        start = 0
        
        # Find all occurrences of the keyword
        while True:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break
                
            # Get the context around the keyword
            context_start = max(0, pos - context_length)
            context_end = min(len(text), pos + len(keyword) + context_length)
            
            # Extract the actual text with original casing
            before = text[context_start:pos]
            keyword_actual = text[pos:pos+len(keyword)]
            after = text[pos+len(keyword):context_end]
            
            # Create a formatted context
            if context_start > 0:
                before = f"...{before}"
            if context_end < len(text):
                after = f"{after}..."
                
            context = f"{before}**{keyword_actual}**{after}"
            matches.append(context)
            
            # Move to the next position
            start = pos + len(keyword)
            
            # Limit to first 3 occurrences to keep the report manageable
            if len(matches) >= 3:
                break
                
        return matches

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
                            combined_text = f"{entry_title} {clean_content}".lower()
                            if keyword.lower() in combined_text:
                                if keyword not in self.results:
                                    self.results[keyword] = []
                                
                                # Get context snippets for the keyword
                                title_context = []
                                if keyword.lower() in entry_title.lower():
                                    title_context = self.get_context(entry_title, keyword)
                                
                                content_context = []
                                if keyword.lower() in clean_content.lower():
                                    content_context = self.get_context(clean_content, keyword)
                                
                                self.results[keyword].append({
                                    "feed": feed_title,
                                    "title": entry_title,
                                    "link": entry_link,
                                    "date": entry_date,
                                    "file": entry_path,
                                    "title_context": title_context,
                                    "content_context": content_context
                                })
            
            except Exception as e:
                print(f"Error processing feed {feed_title}: {e}")
    
    def generate_report(self):
        """Generate a report of the keywords found with context and clickable links."""
        if not self.results:
            print("No keyword matches found.")
            return None, None
        
        # Generate both a text report and an HTML report for clickable links
        timestamp = datetime.now().strftime("%Y-%m-%d")
        text_report_path = os.path.join(self.output_dir, f"keyword_report_{timestamp}.txt")
        html_report_path = os.path.join(self.output_dir, f"keyword_report_{timestamp}.html")
        
        # Generate text report
        with open(text_report_path, "w", encoding="utf-8") as f:
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
                    f.write(f"File: {match['file']}\n")
                    
                    # Add context snippets
                    if match['title_context']:
                        f.write("\nFound in title:\n")
                        for ctx in match['title_context']:
                            f.write(f"  - {ctx}\n")
                    
                    if match['content_context']:
                        f.write("\nFound in content:\n")
                        for ctx in match['content_context']:
                            f.write(f"  - {ctx}\n")
                    
                    f.write("\n")
                
                f.write("\n")
        
        # Generate HTML report with clickable links
        with open(html_report_path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang='en'>\n")
            f.write("<head>\n")
            f.write("  <meta charset='UTF-8'>\n")
            f.write("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
            f.write(f"  <title>Keyword Search Report - {timestamp}</title>\n")
            f.write("  <style>\n")
            f.write("    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }\n")
            f.write("    h1, h2, h3 { color: #333; }\n")
            f.write("    .keyword { color: #2c5aa0; font-weight: bold; }\n")
            f.write("    .match { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 15px; }\n")
            f.write("    .context { margin-left: 20px; background-color: #f9f9f9; padding: 10px; border-left: 3px solid #ddd; }\n")
            f.write("    .highlight { background-color: #ffff00; font-weight: bold; }\n")
            f.write("    a { color: #0066cc; text-decoration: none; }\n")
            f.write("    a:hover { text-decoration: underline; }\n")
            f.write("    .file-link { font-family: monospace; }\n")
            f.write("  </style>\n")
            f.write("</head>\n")
            f.write("<body>\n")
            f.write("  <h1>Keyword Search Report</h1>\n")
            f.write(f"  <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            f.write(f"  <p>Keywords: {', '.join(['<span class=\"keyword\">' + k + '</span>' for k in self.keywords])}</p>\n")
            
            # Create table of contents
            f.write("  <h2>Table of Contents</h2>\n")
            f.write("  <ul>\n")
            for keyword in self.results.keys():
                f.write(f'    <li><a href="#{keyword.replace(" ", "_")}">{keyword} ({len(self.results[keyword])} matches)</a></li>\n')
            f.write("  </ul>\n")
            
            # Add each keyword section
            for keyword, matches in self.results.items():
                f.write(f'  <h2 id="{keyword.replace(" ", "_")}">Keyword: {keyword}</h2>\n')
                f.write(f"  <p>Found in {len(matches)} entries</p>\n")
                
                for i, match in enumerate(matches):
                    f.write(f'  <div class="match">\n')
                    f.write(f'    <h3>{i+1}. <a href="{match["link"]}" target="_blank">{match["title"]}</a></h3>\n')
                    f.write(f'    <p>Feed: {match["feed"]}<br>\n')
                    f.write(f'    Date: {match["date"]}<br>\n')
                    f.write(f'    File: <span class="file-link">{match["file"]}</span></p>\n')
                    
                    # Add context snippets
                    if match['title_context']:
                        f.write('    <p><strong>Found in title:</strong></p>\n')
                        f.write('    <div class="context">\n')
                        for ctx in match['title_context']:
                            # Replace ** markers with HTML highlight spans
                            html_ctx = re.sub(r'\*\*(.*?)\*\*', r'<span class="highlight">\1</span>', ctx)
                            f.write(f'      <p>{html_ctx}</p>\n')
                        f.write('    </div>\n')
                    
                    if match['content_context']:
                        f.write('    <p><strong>Found in content:</strong></p>\n')
                        f.write('    <div class="context">\n')
                        for ctx in match['content_context']:
                            # Replace ** markers with HTML highlight spans
                            html_ctx = re.sub(r'\*\*(.*?)\*\*', r'<span class="highlight">\1</span>', ctx)
                            f.write(f'      <p>{html_ctx}</p>\n')
                        f.write('    </div>\n')
                    
                    f.write('  </div>\n')
                
                f.write('\n')
            
            f.write("</body>\n")
            f.write("</html>\n")
        
        print(f"Text report generated: {text_report_path}")
        print(f"HTML report with clickable links generated: {html_report_path}")
        
        return text_report_path, html_report_path

# Load up the Keywords.txt file
def load_keywords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error loading keywords file: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Process RSS feeds from OPML and search for keywords")
    parser.add_argument("opml_file", help="Path to the OPML file containing RSS feeds")
    parser.add_argument("--output-dir", "-o", default="feed_content", help="Directory to store downloaded content")
    parser.add_argument("--file", "-f", help="Path to a file containing keywords, one per line")
    parser.add_argument("--keywords", "-k", nargs="+", help="Keywords to search for in the feed content")
    parser.add_argument("--max-entries", "-m", type=int, default=10, help="Maximum entries to process per feed")
      
    args = parser.parse_args()
    
    # Collect keywords from both file and command line arguments
    keywords = []
    
    # Read keywords from file if provided
    if args.file:
        file_keywords = load_keywords(args.file)
        keywords.extend(file_keywords)
        print(f"Loaded {len(file_keywords)} keywords from file")
    
    # Add command line keywords if provided
    if args.keywords:
        keywords.extend(args.keywords)
    
    if not keywords:
        print("Warning: No keywords specified. Will download feeds but won't search for keywords.")
    
    processor = RSSProcessor(args.opml_file, args.output_dir, keywords)
    processor.parse_opml()
    processor.download_feeds(args.max_entries)
        
if __name__ == "__main__":
    main()