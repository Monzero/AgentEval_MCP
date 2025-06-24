"""
MCP Question Agent - Intelligent Research Question Generation System
====================================================================

Converted to use MCP tools and A2A communication while maintaining the same functionality
as the original QuestionAgent. This agent generates strategic research questions based on 
topic definitions and rubrics, then generates follow-up questions based on evidence gaps.

MCP Tools Provided:
- generate_initial_question: Create first strategic question from topic/rubric
- generate_follow_up_question: Create follow-up questions based on evidence gaps
- analyze_rubric_requirements: Analyze rubric to identify key differentiators
- assess_evidence_gaps: Identify gaps in collected evidence

A2A Events:
- question_generated: Broadcast when new question is created
- evidence_gap_identified: Broadcast when gap in evidence is found
- questioning_complete: Broadcast when no more questions needed
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import base framework
from mcp_a2a_base import MCPAgent, MCPToolSchema, A2AMessage, print_mcp_action

# Import original dependencies
from main import TopicDefinition, Question, Answer, OptimizedConfig, LLMManager, Colors

logger = logging.getLogger(__name__)

class MCPQuestionAgent(MCPAgent):
    """
    MCP-enabled Question Agent
    
    Provides the same question generation functionality as the original QuestionAgent
    but exposed as MCP tools with A2A communication capabilities.
    """
    
    def __init__(self, config: OptimizedConfig, message_bus):
        # Initialize LLM manager (same as original)
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
        
        # Initialize as MCP agent
        super().__init__("question_agent", config, message_bus)
        
        # Subscribe to relevant events
        self.subscribe_to_event("topic_validation_completed")
        self.subscribe_to_event("research_iteration_completed")
        self.subscribe_to_event("evidence_collected")
    
    def _setup_llm(self):
        """Setup LLM for question generation (same as original)"""
        self.llm, self.current_model = self.llm_manager.get_llm("question_agent")
    
    def _register_tools(self):
        """Register MCP tools for question generation"""
        
        # Initial question generation tool
        self.register_tool(
            MCPToolSchema(
                name="generate_initial_question",
                description="Generate the first strategic question based on topic and rubric analysis",
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
            self._generate_initial_question_tool
        )
        
        # Follow-up question generation tool
        self.register_tool(
            MCPToolSchema(
                name="generate_follow_up_question",
                description="Generate follow-up question based on gaps in existing answers",
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
                        },
                        "existing_answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "answer": {"type": "string"},
                                    "sources": {"type": "array", "items": {"type": "string"}},
                                    "confidence": {"type": "string"},
                                    "has_citations": {"type": "boolean"}
                                }
                            }
                        }
                    },
                    "required": ["topic", "existing_answers"]
                }
            ),
            self._generate_follow_up_question_tool
        )
        
        # Rubric analysis tool
        self.register_tool(
            MCPToolSchema(
                name="analyze_rubric_requirements",
                description="Analyze scoring rubric to identify key differentiators between levels",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rubric": {"type": "object"},
                        "topic_context": {"type": "string"}
                    },
                    "required": ["rubric"]
                }
            ),
            self._analyze_rubric_requirements_tool
        )
        
        # Evidence gap assessment tool
        self.register_tool(
            MCPToolSchema(
                name="assess_evidence_gaps",
                description="Identify gaps in collected evidence that prevent proper scoring",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rubric_requirements": {"type": "array", "items": {"type": "string"}},
                        "existing_evidence": {"type": "array", "items": {"type": "object"}},
                        "topic_guidance": {"type": "string"}
                    },
                    "required": ["rubric_requirements", "existing_evidence"]
                }
            ),
            self._assess_evidence_gaps_tool
        )
    
    async def _generate_initial_question_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Generate initial strategic question
        
        Same logic as original generate_initial_question but as an MCP tool
        """
        topic_data = params["topic"]
        
        # Convert dict to TopicDefinition object
        topic = TopicDefinition(
            topic_name=topic_data["topic_name"],
            goal=topic_data["goal"],
            guidance=topic_data["guidance"],
            scoring_rubric=topic_data["scoring_rubric"]
        )
        
        if not self.llm:
            fallback_question = {
                "text": f"What information is available about {topic.topic_name}?",
                "purpose": "Fallback question due to LLM unavailability",
                "priority": "high"
            }
            print(f"   ⚠️ No LLM available, using fallback question")
            print(f"   ❓ Question: {fallback_question['text']}")
            
            await self.broadcast_event("question_generated", {
                "question": fallback_question["text"],
                "purpose": fallback_question["purpose"],
                "method": "fallback",
                "topic_name": topic.topic_name
            })
            
            return fallback_question
        
        prompt = f"""
        You are an expert corporate governance analyst. Analyze this topic and create ONE key question that will help distinguish between the scoring levels.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        Your task:
        1. Identify what key information differentiates between score levels
        2. Create ONE specific, document-searchable question that targets this differentiator
        3. The question should be answerable using corporate documents
        
        Respond in JSON format:
        {{
            "question": "Your specific question here",
            "purpose": "Why this question helps distinguish between rubric levels",
            "priority": "high"
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
                
                question_result = {
                    "text": result.get("question", ""),
                    "purpose": result.get("purpose", ""),
                    "priority": result.get("priority", "high")
                }
                
                print(f"   ✅ Generated question: {question_result['text']}")
                print(f"   🎯 Purpose: {question_result['purpose']}")
                
                # Broadcast question generated event
                await self.broadcast_event("question_generated", {
                    "question": question_result["text"],
                    "purpose": question_result["purpose"],
                    "priority": question_result["priority"],
                    "method": "llm_generated",
                    "topic_name": topic.topic_name
                })
                
                return question_result
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            fallback_question = {
                "text": f"What specific information about {topic.topic_name} is disclosed in the documents?",
                "purpose": "Fallback question due to parsing error",
                "priority": "high"
            }
            print(f"   ⚠️ Error occurred, using fallback question")
            print(f"   ❓ Question: {fallback_question['text']}")
            
            await self.broadcast_event("question_generated", {
                "question": fallback_question["text"],
                "purpose": fallback_question["purpose"],
                "method": "error_fallback",
                "topic_name": topic.topic_name,
                "error": str(e)
            })
            
            return fallback_question
    
    async def _generate_follow_up_question_tool(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        MCP Tool: Generate follow-up question based on evidence gaps
        
        Same logic as original generate_follow_up_question but as an MCP tool
        """
        topic_data = params["topic"]
        existing_answers_data = params["existing_answers"]
        
        # Convert to objects
        topic = TopicDefinition(
            topic_name=topic_data["topic_name"],
            goal=topic_data["goal"],
            guidance=topic_data["guidance"],
            scoring_rubric=topic_data["scoring_rubric"]
        )
        
        existing_answers = [
            Answer(
                question=ans["question"],
                answer=ans["answer"],
                sources=ans["sources"],
                confidence=ans["confidence"],
                has_citations=ans["has_citations"]
            )
            for ans in existing_answers_data
        ]
        
        if not self.llm:
            print(f"   ⚠️ No LLM available for follow-up questions")
            await self.broadcast_event("questioning_complete", {
                "reason": "no_llm_available",
                "topic_name": topic.topic_name
            })
            return None
        
        answer_context = "\n".join([
            f"Q: {ans.question}\nA: {ans.answer[:200]}...\nSources: {', '.join(ans.sources[:3])}"
            for ans in existing_answers
        ])
        
        prompt = f"""
        You are evaluating a corporate governance topic. Based on existing research, determine if you need ONE more question to properly apply the scoring rubric.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        EXISTING RESEARCH:
        {answer_context}
        
        Analysis:
        1. Can you confidently apply the scoring rubric with the existing information?
        2. What specific gap prevents proper scoring?
        3. If a gap exists, what ONE question would fill it?
        
        Respond in JSON format:
        {{
            "needs_more_info": true/false,
            "gap_identified": "description of information gap",
            "question": "specific question to fill the gap (if needed)",
            "purpose": "how this question enables proper scoring",
            "priority": "high/medium/low"
        }}
        
        Only generate a question if it's truly necessary for scoring. Be conservative.
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
                
                if result.get("needs_more_info", False) and result.get("question"):
                    question_result = {
                        "text": result["question"],
                        "purpose": result.get("purpose", ""),
                        "priority": result.get("priority", "medium")
                    }
                    
                    print(f"   ✅ Follow-up needed: {question_result['text']}")
                    print(f"   🎯 Gap: {result.get('gap_identified', 'Not specified')}")
                    
                    # Broadcast evidence gap identified event
                    await self.broadcast_event("evidence_gap_identified", {
                        "gap_description": result.get("gap_identified", ""),
                        "follow_up_question": question_result["text"],
                        "topic_name": topic.topic_name
                    })
                    
                    return question_result
                else:
                    print(f"   ✅ No follow-up needed - sufficient information for scoring")
                    await self.broadcast_event("questioning_complete", {
                        "reason": "sufficient_evidence",
                        "topic_name": topic.topic_name,
                        "evidence_count": len(existing_answers)
                    })
                    return None
            
            return None
                
        except Exception as e:
            logger.error(f"Follow-up question generation error: {e}")
            print(f"   ⚠️ Error in follow-up analysis")
            await self.broadcast_event("questioning_complete", {
                "reason": "error_in_analysis",
                "topic_name": topic.topic_name,
                "error": str(e)
            })
            return None
    
    async def _analyze_rubric_requirements_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Analyze rubric to identify key requirements
        
        Extracts key differentiators and requirements from scoring rubric
        """
        rubric = params["rubric"]
        topic_context = params.get("topic_context", "")
        
        # Extract key requirements from each rubric level
        requirements = []
        differentiators = []
        
        for level, criteria in rubric.items():
            if criteria and criteria.strip():
                # Extract key terms and concepts
                level_requirements = self._extract_key_concepts(criteria)
                requirements.extend([f"Level {level}: {req}" for req in level_requirements])
                
                # Identify differentiating factors
                if "permanent" in criteria.lower():
                    differentiators.append("director_permanence")
                if "lender" in criteria.lower() or "representative" in criteria.lower():
                    differentiators.append("lender_representation")
                if "disclosure" in criteria.lower():
                    differentiators.append("disclosure_level")
                if "independence" in criteria.lower():
                    differentiators.append("board_independence")
        
        return {
            "requirements": requirements,
            "key_differentiators": list(set(differentiators)),
            "rubric_complexity": len(requirements),
            "analysis_summary": f"Identified {len(requirements)} requirements across {len(rubric)} scoring levels"
        }
    
    async def _assess_evidence_gaps_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Assess gaps in collected evidence
        
        Compares rubric requirements against existing evidence to identify gaps
        """
        rubric_requirements = params["rubric_requirements"]
        existing_evidence = params["existing_evidence"]
        topic_guidance = params.get("topic_guidance", "")
        
        # Analyze evidence coverage
        covered_requirements = []
        missing_requirements = []
        
        # Combine all evidence text for analysis
        all_evidence_text = " ".join([
            evidence.get("answer", "") for evidence in existing_evidence
        ]).lower()
        
        for requirement in rubric_requirements:
            requirement_lower = requirement.lower()
            
            # Check if requirement concepts are mentioned in evidence
            key_terms = self._extract_key_concepts(requirement_lower)
            if any(term in all_evidence_text for term in key_terms):
                covered_requirements.append(requirement)
            else:
                missing_requirements.append(requirement)
        
        # Assess overall completeness
        coverage_ratio = len(covered_requirements) / len(rubric_requirements) if rubric_requirements else 1.0
        
        return {
            "covered_requirements": covered_requirements,
            "missing_requirements": missing_requirements,
            "coverage_ratio": coverage_ratio,
            "evidence_quality": "high" if coverage_ratio > 0.8 else "medium" if coverage_ratio > 0.5 else "low",
            "gaps_identified": len(missing_requirements),
            "recommendation": "sufficient" if coverage_ratio > 0.7 else "needs_more_evidence"
        }
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple keyword extraction
        key_concepts = []
        
        # Common governance terms
        governance_terms = [
            "director", "board", "independent", "permanent", "appointment", "reappointment",
            "lender", "representative", "disclosure", "compensation", "audit", "risk",
            "shareholder", "voting", "transparency", "governance"
        ]
        
        text_lower = text.lower()
        for term in governance_terms:
            if term in text_lower:
                key_concepts.append(term)
        
        return key_concepts
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events from other agents"""
        if event_type == "topic_validation_completed":
            if data.get("valid", False):
                print(f"   📢 Topic validation completed successfully for: {data.get('topic_name', 'Unknown')}")
            else:
                print(f"   📢 Topic validation failed for: {data.get('topic_name', 'Unknown')}")
        
        elif event_type == "research_iteration_completed":
            print(f"   📢 Research iteration {data.get('iteration', 'Unknown')} completed")
        
        elif event_type == "evidence_collected":
            print(f"   📢 New evidence collected from {data.get('sources', 'Unknown')} sources")

# Backward compatibility wrapper
class QuestionAgentWrapper:
    """
    Wrapper to maintain backward compatibility with original interface
    """
    
    def __init__(self, mcp_agent: MCPQuestionAgent):
        self.mcp_agent = mcp_agent
        self.current_model = mcp_agent.current_model
    
    async def generate_initial_question(self, topic: TopicDefinition) -> Question:
        """Original interface method - calls MCP tool internally"""
        result = await self.mcp_agent.call_tool("generate_initial_question", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }
        })
        
        return Question(
            text=result["text"],
            purpose=result["purpose"],
            priority=result["priority"]
        )
    
    async def generate_follow_up_question(self, topic: TopicDefinition, existing_answers: List[Answer]) -> Optional[Question]:
        """Original interface method - calls MCP tool internally"""
        result = await self.mcp_agent.call_tool("generate_follow_up_question", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            },
            "existing_answers": [
                {
                    "question": ans.question,
                    "answer": ans.answer,
                    "sources": ans.sources,
                    "confidence": ans.confidence,
                    "has_citations": ans.has_citations
                }
                for ans in existing_answers
            ]
        })
        
        if result is None:
            return None
        
        return Question(
            text=result["text"],
            purpose=result["purpose"],
            priority=result["priority"]
        )

# Factory function for creating MCP Question Agent
def create_mcp_question_agent(config: OptimizedConfig, message_bus) -> MCPQuestionAgent:
    """Factory function to create MCP Question Agent"""
    return MCPQuestionAgent(config, message_bus)