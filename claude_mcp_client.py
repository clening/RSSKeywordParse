#!/usr/bin/env python3
"""
Claude MCP Client for Palantir Intelligence
Integrates the intelligence processor with Claude MCP for enhanced analysis
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
import logging

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("MCP client not installed. Install with: pip install mcp")
    sys.exit(1)

logger = logging.getLogger(__name__)

class PalantirMCPClient:
    """Client for interacting with Palantir Intelligence MCP server."""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "palantir_mcp_server"],
            env={
                "PALANTIR_KEYWORDS_FILE": os.getenv("PALANTIR_KEYWORDS_FILE", "./keywords.txt"),
                "PALANTIR_OUTPUT_DIR": os.getenv("PALANTIR_OUTPUT_DIR", "./output"),
                "CLAUDE_MCP_ENABLED": "true"
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = await stdio_client(self.server_params).__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def analyze_content(self, content: str, title: str, source: str = "", content_type: str = "news") -> Dict[str, Any]:
        """Analyze content using MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await self.session.call_tool(
                "analyze_palantir_content",
                {
                    "content": content,
                    "title": title,
                    "source": source,
                    "content_type": content_type
                }
            )
            
            return {
                "success": True,
                "analysis": result.content[0].text if result.content else "No analysis available"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": f"Basic analysis: {content[:200]}..."
            }
    
    async def extract_contract_details(self, content: str, document_type: str = "contract_award") -> Dict[str, Any]:
        """Extract contract details using MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await self.session.call_tool(
                "extract_contract_details",
                {
                    "content": content,
                    "document_type": document_type
                }
            )
            
            return {
                "success": True,
                "details": result.content[0].text if result.content else "No details extracted"
            }
            
        except Exception as e:
            logger.error(f"Error extracting contract details: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": "Contract detail extraction failed"
            }
    
    async def identify_data_systems(self, content: str) -> Dict[str, Any]:
        """Identify data systems using MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await self.session.call_tool(
                "identify_data_systems",
                {
                    "content": content
                }
            )
            
            return {
                "success": True,
                "systems": result.content[0].text if result.content else "No systems identified"
            }
            
        except Exception as e:
            logger.error(f"Error identifying data systems: {e}")
            return {
                "success": False,
                "error": str(e),
                "systems": "Data system identification failed"
            }
    
    async def assess_strategic_impact(self, content: str, context: str = "") -> Dict[str, Any]:
        """Assess strategic impact using MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await self.session.call_tool(
                "assess_strategic_impact",
                {
                    "content": content,
                    "context": context
                }
            )
            
            return {
                "success": True,
                "assessment": result.content[0].text if result.content else "No assessment available"
            }
            
        except Exception as e:
            logger.error(f"Error assessing strategic impact: {e}")
            return {
                "success": False,
                "error": str(e),
                "assessment": "Strategic impact assessment failed"
            }
    
    async def generate_executive_summary(self, items: List[Dict]) -> Dict[str, Any]:
        """Generate executive summary using MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            result = await self.session.call_tool(
                "generate_executive_summary",
                {
                    "items": items,
                    "time_period": "recent period"
                }
            )
            
            return {
                "success": True,
                "summary": result.content[0].text if result.content else "No summary generated"
            }
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "Executive summary generation failed"
            }
    
    async def list_available_tools(self) -> List[str]:
        """List available MCP tools."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            tools = await self.session.list_tools()
            return [tool.name for tool in tools]
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []
    
    async def list_available_resources(self) -> List[str]:
        """List available MCP resources."""
        if not self.session:
            raise RuntimeError("MCP session not initialized")
        
        try:
            resources = await self.session.list_resources()
            return [resource.name for resource in resources]
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return []

# Enhanced Intelligence Processor with MCP Integration
class EnhancedPalantirProcessor:
    """Enhanced processor that uses MCP for analysis."""
    
    def __init__(self, base_processor):
        self.base_processor = base_processor
        self.mcp_client = PalantirMCPClient()
    
    async def process_with_mcp(self):
        """Process intelligence items with MCP enhancement."""
        
        # Run base processing first
        await self.base_processor.process_all_sources()
        
        if not self.base_processor.results:
            print("No intelligence items to enhance")
            return
        
        print(f"Enhancing {len(self.base_processor.results)} items with Claude MCP...")
        
        async with self.mcp_client as mcp:
            # List available tools
            tools = await mcp.list_available_tools()
            print(f"Available MCP tools: {', '.join(tools)}")
            
            enhanced_results = {}
            
            for item_id, item in self.base_processor.results.items():
                print(f"Enhancing: {item['title'][:50]}...")
                
                # Basic content analysis
                analysis = await mcp.analyze_content(
                    content=item.get('content', ''),
                    title=item.get('title', ''),
                    source=item.get('source', ''),
                    content_type=item.get('source_type', 'news')
                )
                
                # Contract-specific analysis if applicable
                contract_details = None
                if item.get('source_type') in ['contract', 'federal_register']:
                    contract_details = await mcp.extract_contract_details(
                        content=item.get('content', ''),
                        document_type=item.get('source_type', 'contract_award')
                    )
                
                # Data systems analysis
                data_systems = await mcp.identify_data_systems(
                    content=item.get('content', '')
                )
                
                # Strategic impact assessment
                strategic_impact = await mcp.assess_strategic_impact(
                    content=item.get('content', ''),
                    context=f"Source: {item.get('source', '')}, Type: {item.get('source_type', '')}"
                )
                
                # Enhance the item with MCP analysis
                enhanced_item = item.copy()
                enhanced_item.update({
                    'mcp_analysis': analysis.get('analysis', ''),
                    'mcp_analysis_success': analysis.get('success', False),
                    'contract_details': contract_details.get('details', '') if contract_details else '',
                    'data_systems': data_systems.get('systems', ''),
                    'strategic_impact': strategic_impact.get('assessment', ''),
                    'enhanced': True
                })
                
                enhanced_results[item_id] = enhanced_item
            
            # Generate executive summary
            if enhanced_results:
                summary_items = [
                    {
                        'title': item['title'],
                        'content': item.get('content', ''),
                        'source': item.get('source', ''),
                        'date': item.get('date', '')
                    }
                    for item in enhanced_results.values()
                ]
                
                exec_summary = await mcp.generate_executive_summary(summary_items)
                
                # Store executive summary
                self.base_processor.executive_summary = exec_summary.get('summary', '')
                self.base_processor.executive_summary_success = exec_summary.get('success', False)
            
            # Update base processor results
            self.base_processor.results = enhanced_results
            
            print(f"✅ Enhanced {len(enhanced_results)} items with Claude MCP")
    
    def generate_enhanced_report(self):
        """Generate enhanced HTML report with MCP analysis."""
        return self.base_processor.generate_intelligence_report()

# Example usage and testing
async def test_mcp_integration():
    """Test MCP integration with sample data."""
    
    print("🧪 Testing Claude MCP Integration")
    print("=" * 40)
    
    async with PalantirMCPClient() as mcp:
        # Test tool listing
        tools = await mcp.list_available_tools()
        print(f"Available tools: {tools}")
        
        # Test resource listing
        resources = await mcp.list_available_resources()
        print(f"Available resources: {resources}")
        
        # Test content analysis
        sample_content = """
        Palantir Technologies has been awarded a $90 million contract by the Department of Defense
        to provide data analytics services for intelligence operations. The contract includes
        access to the Gotham platform for analyzing classified datasets and developing
        predictive models for threat assessment.
        """
        
        analysis = await mcp.analyze_content(
            content=sample_content,
            title="Palantir Wins $90M DoD Contract",
            source="Defense News",
            content_type="contract"
        )
        
        print("\n📊 Content Analysis Result:")
        print(analysis.get('analysis', 'No analysis'))
        
        # Test contract details extraction
        contract_details = await mcp.extract_contract_details(
            content=sample_content,
            document_type="contract_award"
        )
        
        print("\n📋 Contract Details:")
        print(contract_details.get('details', 'No details'))
        
        # Test data systems identification
        data_systems = await mcp.identify_data_systems(content=sample_content)
        
        print("\n🗄️ Data Systems:")
        print(data_systems.get('systems', 'No systems'))
        
        # Test strategic assessment
        strategic = await mcp.assess_strategic_impact(
            content=sample_content,
            context="Major DoD contract award"
        )
        
        print("\n🎯 Strategic Impact:")
        print(strategic.get('assessment', 'No assessment'))

def main():
    """Main function for testing."""
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run MCP integration test
        asyncio.run(test_mcp_integration())
    else:
        print("Claude MCP Client for Palantir Intelligence")
        print("Usage:")
        print("  python claude_mcp_client.py test    # Run integration test")
        print("  # Or import and use EnhancedPalantirProcessor in your code")

if __name__ == "__main__":
    main()