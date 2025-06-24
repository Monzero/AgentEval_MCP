"""
MCP Input Guardrail Agent - Topic Definition Validation System
==============================================================

Converted to use MCP tools and A2A communication while maintaining the same functionality
as the original InputGuardrailAgent. This agent validates user-provided topic definitions
before expensive document processing begins.

MCP Tools Provided:
- validate_topic_definition: Main validation tool
- check_rubric_completeness: Rubric-specific validation
- suggest_improvements: Generate improvement suggestions

A2A Events:
- topic_validation_completed: Broadcast when validation finishes
- validation_failed: Broadcast when validation fails
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import base framework
from mcp_a2a_base import MCPAgent, MCPToolSchema, A2AMessage, print_mcp_action

# Import original dependencies
from main import TopicDefinition, OptimizedConfig, LLMManager, Colors

logger = logging.getLogger(__name__)

class MCPInputGuardrailAgent(MCPAgent):
    """
    MCP-enabled Input Guardrail Agent
    
    Provides the same validation functionality as the original InputGuardrailAgent
    but exposed as MCP tools with A2A communication capabilities.
    """
    
    def __init__(self, config: OptimizedConfig, message_bus):
        # Initialize LLM manager (same as original)
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
        
        # Initialize as MCP agent
        super().__init__("input_guardrail", config, message_bus)
        
        # Subscribe to relevant events
        self.subscribe_to_event("system_startup")
        self.subscribe_to_event("evaluation_started")
    
    def _setup_llm(self):
        """Setup LLM for input validation (same as original)"""
        self.llm, self.current_model = self.llm_manager.get_llm("input_agent")
    
    def _register_tools(self):
        """Register MCP tools for input validation"""
        
        # Main validation tool
        self.register_tool(
            MCPToolSchema(
                name="validate_topic_definition",
                description="Validate if topic definition is appropriate for evaluation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "object",
                            "properties": {
                                "topic_name": {"type": "string"},
                                "goal": {"type": "string"},
                                "guidance": {"type": "string"},
                                "scoring_rubric": {"type": "object"}
                            },
                            "required": ["topic_name", "goal", "guidance", "scoring_rubric"]
                        }
                    },
                    "required": ["topic"]
                }
            ),
            self._validate_topic_definition_tool
        )
        
        # Rubric validation tool
        self.register_tool(
            MCPToolSchema(
                name="check_rubric_completeness",
                description="Check if scoring rubric is complete and well-formed",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rubric": {"type": "object"},
                        "expected_levels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["0", "1", "2"]
                        }
                    },
                    "required": ["rubric"]
                }
            ),
            self._check_rubric_completeness_tool
        )
        
        # Improvement suggestions tool
        self.register_tool(
            MCPToolSchema(
                name="suggest_improvements",
                description="Generate suggestions for improving topic definition",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "object"},
                        "validation_issues": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["topic", "validation_issues"]
                }
            ),
            self._suggest_improvements_tool
        )
    
    async def _validate_topic_definition_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Validate topic definition
        
        Same logic as original validate_topic_definition but as an MCP tool
        """
        topic_data = params["topic"]
        
        # Convert dict to TopicDefinition object
        topic = TopicDefinition(
            topic_name=topic_data["topic_name"],
            goal=topic_data["goal"],
            guidance=topic_data["guidance"],
            scoring_rubric=topic_data["scoring_rubric"]
        )
        
        print(f"   🔍 Checking basic requirements...")
        
        # Basic validation first (same as original)
        basic_issues = []
        
        if not topic.topic_name or not topic.topic_name.strip():
            basic_issues.append("Topic name is empty")
        
        if not topic.goal or not topic.goal.strip():
            basic_issues.append("Goal is empty")
        
        if not topic.guidance or not topic.guidance.strip():
            basic_issues.append("Guidance is empty")
        
        if not topic.scoring_rubric or len(topic.scoring_rubric) == 0:
            basic_issues.append("Scoring rubric is empty")
        
        if len(topic.goal.strip()) < 10:
            basic_issues.append("Goal is too short to be meaningful")
        
        if len(topic.guidance.strip()) < 10:
            basic_issues.append("Guidance is too short to be meaningful")
        
        if basic_issues:
            print(f"   ❌ Basic validation failed: {basic_issues}")
            result = {
                "valid": False,
                "issues": basic_issues,
                "suggestions": ["Please provide more detailed information for the empty or very short fields"]
            }
            # Broadcast validation failed event
            await self.broadcast_event("validation_failed", {
                "topic_name": topic.topic_name,
                "issues": basic_issues
            })
            return result
        
        print(f"   ✅ Basic validation passed")
        
        if not self.llm:
            print(f"   ⚠️ No LLM available, being permissive")
            result = {"valid": True, "issues": [], "suggestions": []}
            await self.broadcast_event("topic_validation_completed", {
                "topic_name": topic.topic_name,
                "valid": True,
                "validation_method": "basic_only"
            })
            return result
        
        # Use LLM for more nuanced validation (same as original)
        prompt = f"""
        You are helping validate a corporate governance topic definition. Be VERY LENIENT and permissive.
        Only mark as invalid if there are SERIOUS, OBVIOUS problems that would make evaluation impossible.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        SCORING RUBRIC: {json.dumps(topic.scoring_rubric, indent=2)}
        
        Validation criteria (ONLY mark invalid for serious issues):
        1. Is the goal clear enough to understand what needs to be evaluated?
        2. Does the guidance give some direction on how to evaluate?
        3. Does the scoring rubric have different levels? (i.e 0,1 and 2)
        4. Does it contain any inappropriate language?
        5. Is user asking things which is taboo or not allowed?
        
        IMPORTANT: Err on the side of marking topics as VALID
        
        Respond in JSON format:
        {{
            "valid": true/false,
            "issues": ["only list SERIOUS issues that prevent evaluation"],
            "suggestions": ["gentle suggestions for improvement, not requirements"]
        }}
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Apply lenient policy (same as original)
                if not result.get("valid", True) and not basic_issues:
                    print(f"   🔄 LLM marked invalid, but overriding to valid (lenient policy)")
                    result["valid"] = True
                    if result.get("issues"):
                        result["suggestions"] = result.get("suggestions", []) + [f"Consider: {issue}" for issue in result["issues"]]
                        result["issues"] = []
                
                status = "✅ VALID" if result["valid"] else "❌ INVALID"
                print(f"   {status} - Issues: {len(result.get('issues', []))}, Suggestions: {len(result.get('suggestions', []))}")
                
                # Broadcast validation completed event
                await self.broadcast_event("topic_validation_completed", {
                    "topic_name": topic.topic_name,
                    "valid": result["valid"],
                    "validation_method": "llm_enhanced",
                    "issues_count": len(result.get("issues", [])),
                    "suggestions_count": len(result.get("suggestions", []))
                })
                
                return result
            else:
                print(f"   ⚠️ Could not parse LLM response, defaulting to valid")
                result = {"valid": True, "issues": [], "suggestions": []}
                await self.broadcast_event("topic_validation_completed", {
                    "topic_name": topic.topic_name,
                    "valid": True,
                    "validation_method": "fallback"
                })
                return result
                
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            print(f"   ⚠️ Validation error, defaulting to valid")
            result = {"valid": True, "issues": [], "suggestions": []}
            await self.broadcast_event("topic_validation_completed", {
                "topic_name": topic.topic_name,
                "valid": True,
                "validation_method": "error_fallback",
                "error": str(e)
            })
            return result
    
    async def _check_rubric_completeness_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Check rubric completeness
        
        Validates that the scoring rubric has all expected levels and is well-formed
        """
        rubric = params["rubric"]
        expected_levels = params.get("expected_levels", ["0", "1", "2"])
        
        issues = []
        suggestions = []
        
        # Check for missing levels
        missing_levels = []
        for level in expected_levels:
            if level not in rubric:
                missing_levels.append(level)
        
        if missing_levels:
            issues.append(f"Missing scoring levels: {', '.join(missing_levels)}")
            suggestions.append(f"Please add criteria for scoring levels: {', '.join(missing_levels)}")
        
        # Check for empty criteria
        empty_levels = []
        for level, criteria in rubric.items():
            if not criteria or not criteria.strip():
                empty_levels.append(level)
        
        if empty_levels:
            issues.append(f"Empty criteria for levels: {', '.join(empty_levels)}")
            suggestions.append(f"Please provide criteria for levels: {', '.join(empty_levels)}")
        
        # Check for very short criteria
        short_levels = []
        for level, criteria in rubric.items():
            if criteria and len(criteria.strip()) < 10:
                short_levels.append(level)
        
        if short_levels:
            suggestions.append(f"Consider expanding criteria for levels: {', '.join(short_levels)}")
        
        return {
            "complete": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions,
            "missing_levels": missing_levels,
            "empty_levels": empty_levels
        }
    
    async def _suggest_improvements_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Generate improvement suggestions
        
        Analyzes validation issues and generates specific improvement suggestions
        """
        topic_data = params["topic"]
        validation_issues = params["validation_issues"]
        
        suggestions = []
        
        # Generate specific suggestions based on issues
        for issue in validation_issues:
            if "topic name" in issue.lower():
                suggestions.append("Provide a clear, descriptive topic name that indicates what aspect of corporate governance you want to evaluate")
            
            elif "goal" in issue.lower():
                suggestions.append("Expand the goal to clearly state what you want to measure or assess. Include specific outcomes or metrics if possible")
            
            elif "guidance" in issue.lower():
                suggestions.append("Provide detailed guidance including: what documents to examine, what information to look for, and how to interpret findings")
            
            elif "rubric" in issue.lower():
                suggestions.append("Ensure your scoring rubric has clear criteria for each level (0, 1, 2) that distinguish between different performance levels")
        
        # Generate topic-specific suggestions based on content
        topic_name = topic_data.get("topic_name", "").lower()
        if "board" in topic_name:
            suggestions.append("For board-related topics, consider including criteria about independence, diversity, experience, or meeting frequency")
        
        if "disclosure" in topic_name:
            suggestions.append("For disclosure topics, specify what information should be disclosed, where it should appear, and what level of detail is expected")
        
        if "compensation" in topic_name:
            suggestions.append("For compensation topics, clarify whether you're evaluating disclosure completeness, alignment with performance, or governance processes")
        
        return {
            "suggestions": suggestions,
            "suggested_improvements": len(suggestions),
            "topic_category": self._categorize_topic(topic_data.get("topic_name", ""))
        }
    
    def _categorize_topic(self, topic_name: str) -> str:
        """Categorize topic for targeted suggestions"""
        topic_lower = topic_name.lower()
        
        if any(word in topic_lower for word in ["board", "director", "independence"]):
            return "board_governance"
        elif any(word in topic_lower for word in ["compensation", "remuneration", "pay"]):
            return "executive_compensation"
        elif any(word in topic_lower for word in ["disclosure", "transparency", "reporting"]):
            return "disclosure_practices"
        elif any(word in topic_lower for word in ["audit", "internal control", "risk"]):
            return "audit_and_risk"
        elif any(word in topic_lower for word in ["shareholder", "voting", "rights"]):
            return "shareholder_rights"
        else:
            return "general_governance"
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events from other agents"""
        if event_type == "evaluation_started":
            print(f"   📢 Evaluation started for topic: {data.get('topic_name', 'Unknown')}")
            # Could perform any initialization needed
        elif event_type == "system_startup":
            print(f"   📢 System startup detected, input guardrail agent ready")

# Backward compatibility wrapper
class InputGuardrailAgentWrapper:
    """
    Wrapper to maintain backward compatibility with original interface
    
    This allows existing code to work without changes while using MCP underneath
    """
    
    def __init__(self, mcp_agent: MCPInputGuardrailAgent):
        self.mcp_agent = mcp_agent
        self.current_model = mcp_agent.current_model
    
    async def validate_topic_definition(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Original interface method - calls MCP tool internally"""
        return await self.mcp_agent.call_tool("validate_topic_definition", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }
        })

# Factory function for creating MCP Input Guardrail Agent
def create_mcp_input_guardrail_agent(config: OptimizedConfig, message_bus) -> MCPInputGuardrailAgent:
    """Factory function to create MCP Input Guardrail Agent"""
    return MCPInputGuardrailAgent(config, message_bus)