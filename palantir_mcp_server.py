#!/usr/bin/env python3
"""
Palantir Intelligence MCP Server
Provides Claude MCP tools for enhanced intelligence analysis
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import aiohttp
from datetime import datetime, timedelta
import re

# MCP imports
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.types import (
        Resource, 
        Tool, 
        TextContent, 
        ImageContent, 
        EmbeddedResource,
        LoggingLevel
    )
    import mcp.server.stdio
    import mcp.types as types
except ImportError:
    print("MCP package not installed. Install with: pip install mcp")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("palantir-mcp")

class PalantirMCPServer:
    def __init__(self):
        self.server = Server("palantir-intelligence")
        self.keywords_file = os.getenv("PALANTIR_KEYWORDS_FILE", "./keywords.txt")
        self.output_dir = os.getenv("PALANTIR_OUTPUT_DIR", "./intelligence_output")
        self.setup_handlers()
    
    def setup_handlers(self):
        """Set up MCP server handlers."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available intelligence resources."""
            resources = []
            
            # Keywords resource
            if os.path.exists(self.keywords_file):
                resources.append(
                    Resource(
                        uri=f"file://{self.keywords_file}",
                        name="Palantir Keywords",
                        description="Keywords for monitoring Palantir-related content",
                        mimeType="text/plain"
                    )
                )
            
            # Intelligence reports
            if os.path.exists(self.output_dir):
                for file in os.listdir(self.output_dir):
                    if file.endswith('.html') and 'intelligence_report' in file:
                        resources.append(
                            Resource(
                                uri=f"file://{os.path.join(self.output_dir, file)}",
                                name=f"Intelligence Report: {file}",
                                description="Generated Palantir intelligence report",
                                mimeType="text/html"
                            )
                        )
            
            return resources
        
        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a specific resource."""
            if uri.startswith("file://"):
                file_path = uri[7:]  # Remove "file://" prefix
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    raise Exception(f"Failed to read resource {uri}: {e}")
            else:
                raise Exception(f"Unsupported URI scheme: {uri}")
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="analyze_palantir_content",
                    description="Analyze content for Palantir-related intelligence",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to analyze"
                            },
                            "title": {
                                "type": "string", 
                                "description": "Title of the content"
                            },
                            "source": {
                                "type": "string",
                                "description": "Source of the content"
                            },
                            "content_type": {
                                "type": "string",
                                "enum": ["news", "contract", "federal_register", "earnings", "filing"],
                                "description": "Type of content being analyzed"
                            }
                        },
                        "required": ["content", "title"]
                    }
                ),
                Tool(
                    name="extract_contract_details",
                    description="Extract contract-specific details from government documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Document content to analyze"
                            },
                            "document_type": {
                                "type": "string",
                                "enum": ["contract_award", "rfp", "solicitation", "modification"],
                                "description": "Type of contract document"
                            }
                        },
                        "required": ["content"]
                    }
                ),
                Tool(
                    name="identify_data_systems",
                    description="Identify data systems and databases mentioned in content",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to analyze for data systems"
                            }
                        },
                        "required": ["content"]
                    }
                ),
                Tool(
                    name="assess_strategic_impact",
                    description="Assess strategic implications of Palantir-related developments",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to assess"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context about the development"
                            }
                        },
                        "required": ["content"]
                    }
                ),
                Tool(
                    name="generate_executive_summary",
                    description="Generate executive summary of intelligence findings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "content": {"type": "string"},
                                        "source": {"type": "string"},
                                        "date": {"type": "string"}
                                    }
                                },
                                "description": "Intelligence items to summarize"
                            },
                            "time_period": {
                                "type": "string",
                                "description": "Time period covered"
                            }
                        },
                        "required": ["items"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            
            if name == "analyze_palantir_content":
                return await self.analyze_content(arguments)
            elif name == "extract_contract_details":
                return await self.extract_contract_details(arguments)
            elif name == "identify_data_systems":
                return await self.identify_data_systems(arguments)
            elif name == "assess_strategic_impact":
                return await self.assess_strategic_impact(arguments)
            elif name == "generate_executive_summary":
                return await self.generate_executive_summary(arguments)
            else:
                raise Exception(f"Unknown tool: {name}")
    
    async def analyze_content(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Analyze content for Palantir intelligence."""
        content = args.get("content", "")
        title = args.get("title", "")
        source = args.get("source", "")
        content_type = args.get("content_type", "news")
        
        # Load keywords
        keywords = self.load_keywords()
        
        analysis = {
            "summary": self.generate_content_summary(content),
            "palantir_relevance": self.assess_palantir_relevance(content, title),
            "key_entities": self.extract_entities(content),
            "matched_keywords": self.find_matched_keywords(content + " " + title, keywords),
            "sentiment": self.assess_sentiment(content),
            "urgency_level": self.assess_urgency(content, content_type),
            "actionable_insights": self.extract_insights(content, content_type)
        }
        
        # Format analysis
        formatted_analysis = self.format_analysis(analysis, title, source)
        
        return [types.TextContent(type="text", text=formatted_analysis)]
    
    async def extract_contract_details(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Extract contract-specific details."""
        content = args.get("content", "")
        doc_type = args.get("document_type", "contract_award")
        
        details = {
            "contract_value": self.extract_contract_values(content),
            "agencies": self.extract_agencies(content),
            "timeframes": self.extract_timeframes(content),
            "scope_of_work": self.extract_scope(content),
            "data_access": self.extract_data_access(content),
            "security_clearance": self.extract_security_requirements(content),
            "competitors": self.extract_competitors(content)
        }
        
        formatted_details = self.format_contract_details(details, doc_type)
        
        return [types.TextContent(type="text", text=formatted_details)]
    
    async def identify_data_systems(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Identify data systems mentioned in content."""
        content = args.get("content", "")
        
        systems = {
            "databases": self.extract_databases(content),
            "analytics_platforms": self.extract_analytics_platforms(content),
            "data_types": self.extract_data_types(content),
            "access_levels": self.extract_access_levels(content),
            "integration_points": self.extract_integrations(content)
        }
        
        formatted_systems = self.format_data_systems(systems)
        
        return [types.TextContent(type="text", text=formatted_systems)]
    
    async def assess_strategic_impact(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Assess strategic implications."""
        content = args.get("content", "")
        context = args.get("context", "")
        
        assessment = {
            "market_impact": self.assess_market_impact(content),
            "competitive_position": self.assess_competitive_position(content),
            "government_relations": self.assess_gov_relations(content),
            "technology_implications": self.assess_tech_implications(content),
            "risk_factors": self.identify_risks(content),
            "opportunities": self.identify_opportunities(content),
            "recommendations": self.generate_recommendations(content, context)
        }
        
        formatted_assessment = self.format_strategic_assessment(assessment)
        
        return [types.TextContent(type="text", text=formatted_assessment)]
    
    async def generate_executive_summary(self, args: Dict[str, Any]) -> List[types.TextContent]:
        """Generate executive summary."""
        items = args.get("items", [])
        time_period = args.get("time_period", "recent period")
        
        if not items:
            return [types.TextContent(type="text", text="No intelligence items provided for summary.")]
        
        summary = {
            "overview": self.create_overview(items, time_period),
            "key_developments": self.identify_key_developments(items),
            "trend_analysis": self.analyze_trends(items),
            "priority_actions": self.recommend_priority_actions(items),
            "monitoring_recommendations": self.recommend_monitoring(items)
        }
        
        formatted_summary = self.format_executive_summary(summary, len(items), time_period)
        
        return [types.TextContent(type="text", text=formatted_summary)]
    
    # Helper methods for content analysis
    
    def load_keywords(self) -> List[str]:
        """Load keywords from file."""
        try:
            with open(self.keywords_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception:
            return ["Palantir", "PLTR", "Gotham", "Foundry", "AIP", "Alex Karp"]
    
    def generate_content_summary(self, content: str) -> str:
        """Generate a concise summary of content."""
        sentences = re.split(r'[.!?]+', content)
        important_sentences = []
        
        keywords = ["palantir", "contract", "award", "government", "data", "analytics"]
        
        for sentence in sentences[:10]:  # First 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and any(kw in sentence.lower() for kw in keywords):
                important_sentences.append(sentence)
                if len(important_sentences) >= 3:
                    break
        
        return ". ".join(important_sentences) + "." if important_sentences else content[:200] + "..."
    
    def assess_palantir_relevance(self, content: str, title: str) -> Dict[str, Any]:
        """Assess how relevant content is to Palantir."""
        text = f"{title} {content}".lower()
        
        # Direct mentions
        direct_mentions = text.count("palantir")
        
        # Product mentions
        products = ["gotham", "foundry", "aip", "apollo"]
        product_mentions = sum(text.count(product) for product in products)
        
        # Leadership mentions
        leaders = ["alex karp", "shyam sankar", "stephen cohen"]
        leader_mentions = sum(text.count(leader) for leader in leaders)
        
        # Context indicators
        contexts = ["government contract", "defense", "intelligence", "analytics platform"]
        context_score = sum(1 for ctx in contexts if ctx in text)
        
        relevance_score = (direct_mentions * 3) + (product_mentions * 2) + (leader_mentions * 2) + context_score
        
        return {
            "score": min(relevance_score, 10),  # Cap at 10
            "direct_mentions": direct_mentions,
            "product_mentions": product_mentions,
            "leader_mentions": leader_mentions,
            "context_score": context_score
        }
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract key entities from content."""
        entities = {
            "organizations": self.extract_organizations(content),
            "people": self.extract_people(content),
            "locations": self.extract_locations(content),
            "amounts": self.extract_amounts(content)
        }
        return entities
    
    def extract_organizations(self, content: str) -> List[str]:
        """Extract organization names."""
        patterns = [
            r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Agency|Department|Office|Bureau)))\b',
            r'\b(Department of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(U\.S\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        organizations = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            organizations.update(matches)
        
        return list(organizations)[:10]  # Limit results
    
    def extract_people(self, content: str) -> List[str]:
        """Extract person names."""
        # Simple pattern for names (can be enhanced)
        pattern = r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b'
        matches = re.findall(pattern, content)
        
        # Filter out common false positives
        false_positives = {"United States", "New York", "White House", "Supreme Court"}
        people = [name for name in matches if name not in false_positives]
        
        return people[:10]  # Limit results
    
    def extract_locations(self, content: str) -> List[str]:
        """Extract location names."""
        # Simple pattern for locations
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|County|State|Province|Country))?)\b'
        matches = re.findall(pattern, content)
        
        # Common locations
        common_locations = ["Washington", "California", "Virginia", "Maryland", "Pentagon", "Capitol Hill"]
        locations = [loc for loc in matches if loc in common_locations]
        
        return locations[:5]
    
    def extract_amounts(self, content: str) -> List[str]:
        """Extract monetary amounts."""
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?\s*(?:million|billion|thousand|M|B|K)?',
            r'[\d,]+(?:\.\d{2})?\s*(?:million|billion|thousand)\s*dollars?'
        ]
        
        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            amounts.extend(matches)
        
        return amounts[:5]
    
    def find_matched_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords match in the text."""
        text_lower = text.lower()
        matched = []
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        
        return matched
    
    def assess_sentiment(self, content: str) -> str:
        """Basic sentiment assessment."""
        positive_words = ["award", "win", "success", "expand", "grow", "improve", "advance"]
        negative_words = ["lose", "fail", "decline", "issue", "problem", "concern", "risk"]
        
        content_lower = content.lower()
        
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        if pos_count > neg_count + 1:
            return "Positive"
        elif neg_count > pos_count + 1:
            return "Negative"
        else:
            return "Neutral"
    
    def assess_urgency(self, content: str, content_type: str) -> str:
        """Assess urgency level of content."""
        urgent_indicators = ["breaking", "urgent", "immediate", "emergency", "critical"]
        time_indicators = ["today", "this week", "deadline", "expires"]
        
        content_lower = content.lower()
        
        if any(indicator in content_lower for indicator in urgent_indicators):
            return "High"
        elif any(indicator in content_lower for indicator in time_indicators):
            return "Medium"
        elif content_type in ["contract", "federal_register"]:
            return "Medium"
        else:
            return "Low"
    
    def extract_insights(self, content: str, content_type: str) -> List[str]:
        """Extract actionable insights."""
        insights = []
        
        if "contract" in content_type:
            if "award" in content.lower():
                insights.append("Monitor contract performance and deliverables")
            if "bid" in content.lower() or "proposal" in content.lower():
                insights.append("Track competitive bidding process")
        
        if "expand" in content.lower() or "new" in content.lower():
            insights.append("Potential market expansion opportunity")
        
        if "data" in content.lower() and "access" in content.lower():
            insights.append("Data access and integration requirements identified")
        
        return insights
    
    # Additional extraction methods for contract details
    
    def extract_contract_values(self, content: str) -> List[str]:
        """Extract contract values."""
        return self.extract_amounts(content)
    
    def extract_agencies(self, content: str) -> List[str]:
        """Extract government agencies."""
        agencies = []
        agency_patterns = [
            r'\b(Department of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b(DOD|DoD|Pentagon|CIA|FBI|NSA|DHS|ICE|CBP)\b',
            r'\b([A-Z]{2,4})\b(?=\s+(?:agency|department|office))',
        ]
        
        for pattern in agency_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            agencies.extend(matches)
        
        return list(set(agencies))
    
    def extract_timeframes(self, content: str) -> List[str]:
        """Extract timeline information."""
        timeframe_patterns = [
            r'(?:by|until|through|expires?)\s+(\d{4})',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'([A-Z][a-z]+\s+\d{4})',
            r'(\d+\s+(?:year|month|day)s?)'
        ]
        
        timeframes = []
        for pattern in timeframe_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            timeframes.extend(matches)
        
        return list(set(timeframes))
    
    def extract_scope(self, content: str) -> str:
        """Extract scope of work."""
        # Look for scope indicators
        scope_indicators = ["scope", "work", "services", "deliverables", "requirements"]
        
        sentences = re.split(r'[.!?]+', content)
        scope_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in scope_indicators):
                scope_sentences.append(sentence.strip())
                if len(scope_sentences) >= 2:
                    break
        
        return ". ".join(scope_sentences) + "." if scope_sentences else "Scope not clearly defined"
    
    def extract_data_access(self, content: str) -> List[str]:
        """Extract data access information."""
        data_indicators = ["database", "data", "system", "platform", "analytics", "intelligence"]
        
        sentences = re.split(r'[.!?]+', content)
        data_sentences = []
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in data_indicators):
                data_sentences.append(sentence.strip())
        
        return data_sentences[:3]
    
    def extract_security_requirements(self, content: str) -> List[str]:
        """Extract security clearance requirements."""
        security_patterns = [
            r'(Top Secret|Secret|Confidential)(?:\s+clearance)?',
            r'security clearance',
            r'(TS/SCI|TS|S)',
            r'background investigation'
        ]
        
        requirements = []
        for pattern in security_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            requirements.extend(matches)
        
        return list(set(requirements))
    
    def extract_competitors(self, content: str) -> List[str]:
        """Extract mentioned competitors."""
        competitors = ["Microsoft", "Amazon", "Google", "IBM", "Oracle", "Snowflake", "Databricks"]
        
        found_competitors = []
        content_lower = content.lower()
        
        for competitor in competitors:
            if competitor.lower() in content_lower:
                found_competitors.append(competitor)
        
        return found_competitors
    
    # Data system extraction methods
    
    def extract_databases(self, content: str) -> List[str]:
        """Extract database mentions."""
        db_patterns = [
            r'(\w+\s+database)',
            r'(\w+\s+DB)',
            r'(SQL Server|Oracle|PostgreSQL|MySQL|MongoDB)'
        ]
        
        databases = []
        for pattern in db_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            databases.extend(matches)
        
        return list(set(databases))
    
    def extract_analytics_platforms(self, content: str) -> List[str]:
        """Extract analytics platform mentions."""
        platforms = ["Tableau", "Power BI", "Qlik", "Looker", "Databricks", "Snowflake"]
        
        found_platforms = []
        content_lower = content.lower()
        
        for platform in platforms:
            if platform.lower() in content_lower:
                found_platforms.append(platform)
        
        return found_platforms
    
    def extract_data_types(self, content: str) -> List[str]:
        """Extract data types mentioned."""
        data_types = [
            "biometric", "financial", "intelligence", "surveillance", 
            "geospatial", "behavioral", "social media", "communications"
        ]
        
        found_types = []
        content_lower = content.lower()
        
        for data_type in data_types:
            if data_type in content_lower:
                found_types.append(data_type)
        
        return found_types
    
    def extract_access_levels(self, content: str) -> List[str]:
        """Extract access level information."""
        access_patterns = [
            r'(read-only|write|admin|full access)',
            r'(restricted|unrestricted|limited) access',
            r'(public|private|classified) data'
        ]
        
        access_levels = []
        for pattern in access_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            access_levels.extend(matches)
        
        return list(set(access_levels))
    
    def extract_integrations(self, content: str) -> List[str]:
        """Extract integration points."""
        integration_keywords = ["API", "interface", "connector", "integration", "feed"]
        
        sentences = re.split(r'[.!?]+', content)
        integration_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in integration_keywords):
                integration_sentences.append(sentence.strip())
        
        return integration_sentences[:3]
    
    # Strategic assessment methods
    
    def assess_market_impact(self, content: str) -> str:
        """Assess market impact."""
        positive_indicators = ["expansion", "growth", "new market", "increased"]
        negative_indicators = ["decline", "loss", "reduction", "exit"]
        
        content_lower = content.lower()
        
        pos_score = sum(1 for indicator in positive_indicators if indicator in content_lower)
        neg_score = sum(1 for indicator in negative_indicators if indicator in content_lower)
        
        if pos_score > neg_score:
            return "Positive market impact expected"
        elif neg_score > pos_score:
            return "Potential negative market impact"
        else:
            return "Neutral market impact"
    
    def assess_competitive_position(self, content: str) -> str:
        """Assess competitive position."""
        competitive_indicators = ["competition", "competitor", "rival", "alternative"]
        advantage_indicators = ["leading", "preferred", "selected", "chosen", "winner"]
        
        content_lower = content.lower()
        
        competition_mentioned = any(indicator in content_lower for indicator in competitive_indicators)
        advantage_mentioned = any(indicator in content_lower for indicator in advantage_indicators)
        
        if advantage_mentioned and competition_mentioned:
            return "Strong competitive position - preferred vendor"
        elif advantage_mentioned:
            return "Favorable position indicated"
        elif competition_mentioned:
            return "Competitive environment - monitor closely"
        else:
            return "Competitive position unclear"
    
    def assess_gov_relations(self, content: str) -> str:
        """Assess government relations."""
        positive_gov = ["partnership", "collaboration", "trusted", "strategic"]
        negative_gov = ["investigation", "scrutiny", "concern", "violation"]
        
        content_lower = content.lower()
        
        pos_score = sum(1 for indicator in positive_gov if indicator in content_lower)
        neg_score = sum(1 for indicator in negative_gov if indicator in content_lower)
        
        if pos_score > neg_score:
            return "Positive government relations"
        elif neg_score > pos_score:
            return "Potential government relations challenges"
        else:
            return "Stable government relations"
    
    def assess_tech_implications(self, content: str) -> str:
        """Assess technology implications."""
        tech_advancement = ["AI", "machine learning", "innovation", "breakthrough"]
        tech_challenges = ["outdated", "legacy", "limitation", "technical debt"]
        
        content_lower = content.lower()
        
        advancement_score = sum(1 for term in tech_advancement if term in content_lower)
        challenge_score = sum(1 for term in tech_challenges if term in content_lower)
        
        if advancement_score > challenge_score:
            return "Technology advancement opportunities"
        elif challenge_score > advancement_score:
            return "Technology challenges identified"
        else:
            return "Technology status quo"
    
    def identify_risks(self, content: str) -> List[str]:
        """Identify risk factors."""
        risk_indicators = [
            ("regulatory", ["regulation", "compliance", "audit"]),
            ("competitive", ["competitor", "alternative", "threat"]),
            ("technical", ["security", "breach", "vulnerability"]),
            ("financial", ["budget", "cost", "funding"]),
            ("operational", ["delay", "disruption", "issue"])
        ]
        
        risks = []
        content_lower = content.lower()
        
        for risk_type, indicators in risk_indicators:
            if any(indicator in content_lower for indicator in indicators):
                risks.append(f"{risk_type.title()} risk identified")
        
        return risks
    
    def identify_opportunities(self, content: str) -> List[str]:
        """Identify opportunities."""
        opportunity_indicators = [
            ("market_expansion", ["new market", "expansion", "growth"]),
            ("technology", ["innovation", "AI", "automation"]),
            ("partnerships", ["partnership", "collaboration", "alliance"]),
            ("data_access", ["data", "integration", "platform"])
        ]
        
        opportunities = []
        content_lower = content.lower()
        
        for opp_type, indicators in opportunity_indicators:
            if any(indicator in content_lower for indicator in indicators):
                opportunities.append(f"{opp_type.replace('_', ' ').title()} opportunity")
        
        return opportunities
    
    def generate_recommendations(self, content: str, context: str) -> List[str]:
        """Generate strategic recommendations."""
        recommendations = []
        
        content_lower = f"{content} {context}".lower()
        
        if "contract" in content_lower and "award" in content_lower:
            recommendations.append("Monitor contract execution and deliverable milestones")
        
        if "competition" in content_lower:
            recommendations.append("Conduct competitive analysis and positioning review")
        
        if "data" in content_lower:
            recommendations.append("Assess data integration and security requirements")
        
        if "government" in content_lower:
            recommendations.append("Strengthen government stakeholder relationships")
        
        if not recommendations:
            recommendations.append("Continue monitoring for strategic developments")
        
        return recommendations
    
    # Executive summary methods
    
    def create_overview(self, items: List[Dict], time_period: str) -> str:
        """Create overview section."""
        total_items = len(items)
        
        # Categorize items
        contracts = sum(1 for item in items if "contract" in item.get("content", "").lower())
        news = sum(1 for item in items if item.get("source", "").lower() in ["news", "rss"])
        government = sum(1 for item in items if "federal" in item.get("source", "").lower())
        
        overview = f"During the {time_period}, {total_items} intelligence items were collected and analyzed. "
        overview += f"This includes {contracts} contract-related items, {news} news articles, "
        overview += f"and {government} government documents. "
        
        return overview
    
    def identify_key_developments(self, items: List[Dict]) -> List[str]:
        """Identify key developments."""
        developments = []
        
        # Look for high-impact keywords
        high_impact_keywords = ["award", "contract", "selected", "win", "partnership"]
        
        for item in items:
            content = f"{item.get('title', '')} {item.get('content', '')}".lower()
            if any(keyword in content for keyword in high_impact_keywords):
                developments.append(f"• {item.get('title', 'Untitled item')}")
                if len(developments) >= 5:  # Limit to top 5
                    break
        
        return developments
    
    def analyze_trends(self, items: List[Dict]) -> str:
        """Analyze trends across items."""
        # Count mentions of key themes
        themes = {
            "AI/Analytics": ["ai", "analytics", "machine learning", "artificial intelligence"],
            "Government Contracts": ["contract", "award", "government", "federal"],
            "Data Platforms": ["data", "platform", "integration", "database"],
            "Security": ["security", "clearance", "classified", "intelligence"]
        }
        
        theme_counts = {}
        total_content = " ".join([f"{item.get('title', '')} {item.get('content', '')}" 
                                 for item in items]).lower()
        
        for theme, keywords in themes.items():
            count = sum(total_content.count(keyword) for keyword in keywords)
            theme_counts[theme] = count
        
        # Find dominant themes
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if top_themes:
            trend_analysis = "Key trends identified: "
            trend_analysis += ", ".join([f"{theme} ({count} mentions)" 
                                       for theme, count in top_themes if count > 0])
        else:
            trend_analysis = "No clear trends identified in the current dataset."
        
        return trend_analysis
    
    def recommend_priority_actions(self, items: List[Dict]) -> List[str]:
        """Recommend priority actions."""
        actions = []
        
        # Analyze urgency and importance
        urgent_items = [item for item in items if "urgent" in f"{item.get('title', '')} {item.get('content', '')}".lower()]
        contract_items = [item for item in items if "contract" in f"{item.get('title', '')} {item.get('content', '')}".lower()]
        
        if urgent_items:
            actions.append("Address urgent items requiring immediate attention")
        
        if contract_items:
            actions.append("Review contract opportunities and competitive positioning")
        
        if len(items) > 10:
            actions.append("Prioritize high-impact intelligence items for detailed analysis")
        
        actions.append("Continue systematic monitoring of identified sources")
        
        return actions
    
    def recommend_monitoring(self, items: List[Dict]) -> List[str]:
        """Recommend monitoring activities."""
        recommendations = []
        
        # Source diversity
        sources = set(item.get("source", "") for item in items)
        if len(sources) < 3:
            recommendations.append("Expand source diversity for comprehensive coverage")
        
        # Government focus
        gov_items = [item for item in items if "government" in f"{item.get('source', '')} {item.get('content', '')}".lower()]
        if len(gov_items) < len(items) * 0.3:
            recommendations.append("Increase government source monitoring")
        
        # Frequency
        recommendations.append("Maintain daily monitoring schedule for timely intelligence")
        recommendations.append("Set up alerts for high-priority keywords and sources")
        
        return recommendations
    
    # Formatting methods
    
    def format_analysis(self, analysis: Dict, title: str, source: str) -> str:
        """Format content analysis."""
        formatted = f"# Intelligence Analysis: {title}\n\n"
        formatted += f"**Source:** {source}\n\n"
        
        formatted += f"## Summary\n{analysis['summary']}\n\n"
        
        relevance = analysis['palantir_relevance']
        formatted += f"## Palantir Relevance Score: {relevance['score']}/10\n"
        formatted += f"- Direct mentions: {relevance['direct_mentions']}\n"
        formatted += f"- Product mentions: {relevance['product_mentions']}\n"
        formatted += f"- Leadership mentions: {relevance['leader_mentions']}\n\n"
        
        if analysis['matched_keywords']:
            formatted += f"## Matched Keywords\n"
            formatted += ", ".join(analysis['matched_keywords']) + "\n\n"
        
        if analysis['key_entities']['organizations']:
            formatted += f"## Key Organizations\n"
            formatted += ", ".join(analysis['key_entities']['organizations']) + "\n\n"
        
        if analysis['key_entities']['amounts']:
            formatted += f"## Financial Information\n"
            formatted += ", ".join(analysis['key_entities']['amounts']) + "\n\n"
        
        formatted += f"## Assessment\n"
        formatted += f"- **Sentiment:** {analysis['sentiment']}\n"
        formatted += f"- **Urgency:** {analysis['urgency_level']}\n\n"
        
        if analysis['actionable_insights']:
            formatted += f"## Actionable Insights\n"
            for insight in analysis['actionable_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def format_contract_details(self, details: Dict, doc_type: str) -> str:
        """Format contract details."""
        formatted = f"# Contract Analysis ({doc_type.replace('_', ' ').title()})\n\n"
        
        if details['contract_value']:
            formatted += f"## Contract Value\n"
            formatted += ", ".join(details['contract_value']) + "\n\n"
        
        if details['agencies']:
            formatted += f"## Agencies Involved\n"
            formatted += ", ".join(details['agencies']) + "\n\n"
        
        if details['timeframes']:
            formatted += f"## Timeframes\n"
            formatted += ", ".join(details['timeframes']) + "\n\n"
        
        formatted += f"## Scope of Work\n{details['scope_of_work']}\n\n"
        
        if details['data_access']:
            formatted += f"## Data Access Requirements\n"
            for access in details['data_access']:
                formatted += f"- {access}\n"
            formatted += "\n"
        
        if details['security_clearance']:
            formatted += f"## Security Requirements\n"
            formatted += ", ".join(details['security_clearance']) + "\n\n"
        
        if details['competitors']:
            formatted += f"## Potential Competitors\n"
            formatted += ", ".join(details['competitors']) + "\n\n"
        
        return formatted
    
    def format_data_systems(self, systems: Dict) -> str:
        """Format data systems analysis."""
        formatted = "# Data Systems Analysis\n\n"
        
        if systems['databases']:
            formatted += f"## Databases Mentioned\n"
            for db in systems['databases']:
                formatted += f"- {db}\n"
            formatted += "\n"
        
        if systems['analytics_platforms']:
            formatted += f"## Analytics Platforms\n"
            formatted += ", ".join(systems['analytics_platforms']) + "\n\n"
        
        if systems['data_types']:
            formatted += f"## Data Types\n"
            formatted += ", ".join(systems['data_types']) + "\n\n"
        
        if systems['access_levels']:
            formatted += f"## Access Levels\n"
            formatted += ", ".join(systems['access_levels']) + "\n\n"
        
        if systems['integration_points']:
            formatted += f"## Integration Points\n"
            for integration in systems['integration_points']:
                formatted += f"- {integration}\n"
        
        return formatted
    
    def format_strategic_assessment(self, assessment: Dict) -> str:
        """Format strategic assessment."""
        formatted = "# Strategic Impact Assessment\n\n"
        
        formatted += f"## Market Impact\n{assessment['market_impact']}\n\n"
        formatted += f"## Competitive Position\n{assessment['competitive_position']}\n\n"
        formatted += f"## Government Relations\n{assessment['government_relations']}\n\n"
        formatted += f"## Technology Implications\n{assessment['technology_implications']}\n\n"
        
        if assessment['risk_factors']:
            formatted += f"## Risk Factors\n"
            for risk in assessment['risk_factors']:
                formatted += f"- {risk}\n"
            formatted += "\n"
        
        if assessment['opportunities']:
            formatted += f"## Opportunities\n"
            for opp in assessment['opportunities']:
                formatted += f"- {opp}\n"
            formatted += "\n"
        
        if assessment['recommendations']:
            formatted += f"## Strategic Recommendations\n"
            for rec in assessment['recommendations']:
                formatted += f"- {rec}\n"
        
        return formatted
    
    def format_executive_summary(self, summary: Dict, item_count: int, time_period: str) -> str:
        """Format executive summary."""
        formatted = f"# Executive Intelligence Summary\n\n"
        formatted += f"**Period:** {time_period}\n"
        formatted += f"**Items Analyzed:** {item_count}\n\n"
        
        formatted += f"## Overview\n{summary['overview']}\n\n"
        
        if summary['key_developments']:
            formatted += f"## Key Developments\n"
            for dev in summary['key_developments']:
                formatted += f"{dev}\n"
            formatted += "\n"
        
        formatted += f"## Trend Analysis\n{summary['trend_analysis']}\n\n"
        
        if summary['priority_actions']:
            formatted += f"## Priority Actions\n"
            for action in summary['priority_actions']:
                formatted += f"- {action}\n"
            formatted += "\n"
        
        if summary['monitoring_recommendations']:
            formatted += f"## Monitoring Recommendations\n"
            for rec in summary['monitoring_recommendations']:
                formatted += f"- {rec}\n"
        
        return formatted

async def main():
    """Main entry point for the MCP server."""
    server = PalantirMCPServer()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="palantir-intelligence",
                server_version="1.0.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())