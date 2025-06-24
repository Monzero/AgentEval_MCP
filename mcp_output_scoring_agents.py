"""
MCP Output Guardrail and Scoring Agents
=======================================

Converted to use MCP tools and A2A communication while maintaining the same functionality
as the original OutputGuardrailAgent and ScoringAgent.

OutputGuardrailAgent: Validates research answers before scoring
ScoringAgent: Provides final evaluation and scoring based on evidence
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import base framework
from mcp_a2a_base import MCPAgent, MCPToolSchema, A2AMessage, print_mcp_action

# Import original dependencies
from main import TopicDefinition, Answer, OptimizedConfig, LLMManager, Colors

logger = logging.getLogger(__name__)

class MCPOutputGuardrailAgent(MCPAgent):
    """
    MCP-enabled Output Guardrail Agent
    
    Provides the same answer validation functionality as the original OutputGuardrailAgent
    but exposed as MCP tools. This agent uses NO LLM - just deterministic validation logic.
    """
    
    def __init__(self, config: OptimizedConfig, message_bus):
        # Initialize as MCP agent (no LLM needed)
        super().__init__("output_guardrail", config, message_bus)
        
        # Subscribe to relevant events
        self.subscribe_to_event("research_completed")
        self.subscribe_to_event("answer_validation_requested")
    
    def _register_tools(self):
        """Register MCP tools for answer validation"""
        
        # Main validation tool
        self.register_tool(
            MCPToolSchema(
                name="validate_answer",
                description="Validate answer quality and source citations using rule-based logic",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "sources": {"type": "array", "items": {"type": "string"}},
                                "confidence": {"type": "string"},
                                "has_citations": {"type": "boolean"}
                            },
                            "required": ["question", "answer", "sources", "confidence", "has_citations"]
                        }
                    },
                    "required": ["answer"]
                }
            ),
            self._validate_answer_tool
        )
        
        # Batch validation tool
        self.register_tool(
            MCPToolSchema(
                name="validate_answer_batch",
                description="Validate multiple answers at once",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answers": {
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
                    "required": ["answers"]
                }
            ),
            self._validate_answer_batch_tool
        )
        
        # Quality metrics tool
        self.register_tool(
            MCPToolSchema(
                name="assess_answer_quality",
                description="Assess detailed quality metrics for an answer",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answer_text": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "expected_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["length", "citations", "specificity", "completeness"]
                        }
                    },
                    "required": ["answer_text", "sources"]
                }
            ),
            self._assess_answer_quality_tool
        )
    
    async def _validate_answer_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Validate answer quality
        
        Same logic as original validate_answer but as an MCP tool
        """
        answer_data = params["answer"]
        
        # Convert dict to Answer object
        answer = Answer(
            question=answer_data["question"],
            answer=answer_data["answer"],
            sources=answer_data["sources"],
            confidence=answer_data["confidence"],
            has_citations=answer_data["has_citations"]
        )
        
        validation_result = {
            "has_answer": len(answer.answer.strip()) > 20,
            "has_citations": answer.has_citations,
            "confidence": answer.confidence,
            "sources_count": len(answer.sources),
            "retrieval_method": self.config.retrieval_method,
            "issues": [],
            "approved": False
        }
        
        print(f"   📏 Answer length: {len(answer.answer)} characters")
        print(f"   📚 Sources: {validation_result['sources_count']}")
        print(f"   📋 Has citations: {validation_result['has_citations']}")
        print(f"   🎯 Confidence: {validation_result['confidence']}")
        
        # Check for substantive answer
        if not validation_result["has_answer"]:
            validation_result["issues"].append("Answer is too short or empty")
            print(f"   ❌ Answer too short")
        
        # Check for citations
        if not validation_result["has_citations"]:
            validation_result["issues"].append("Answer lacks source citations")
            print(f"   ❌ Missing citations")
        
        # Check confidence
        if answer.confidence == "low":
            validation_result["issues"].append("Low confidence in answer quality")
            print(f"   ⚠️ Low confidence")
        
        # Validation based on retrieval method
        if self.config.retrieval_method == "direct":
            validation_result["approved"] = (
                validation_result["has_answer"] and 
                validation_result["has_citations"]
            )
        elif self.config.retrieval_method in ["hybrid", "bm25", "vector"]:
            if validation_result["sources_count"] < 2:
                validation_result["issues"].append(f"Limited source coverage for {self.config.retrieval_method} retrieval")
                print(f"   ⚠️ Limited source coverage")
            
            validation_result["approved"] = (
                validation_result["has_answer"] and 
                validation_result["has_citations"] and 
                validation_result["sources_count"] >= 1 and
                len(validation_result["issues"]) <= 1
            )
        
        status = "✅ APPROVED" if validation_result["approved"] else "❌ REJECTED"
        print(f"   {status} - Issues: {len(validation_result['issues'])}")
        
        if validation_result["issues"]:
            for issue in validation_result["issues"]:
                print(f"     • {issue}")
        
        # Broadcast validation result
        await self.broadcast_event("answer_validated", {
            "question": answer.question[:100],
            "approved": validation_result["approved"],
            "issues_count": len(validation_result["issues"]),
            "confidence": answer.confidence
        })
        
        return validation_result
    
    async def _validate_answer_batch_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Validate multiple answers at once
        
        Provides batch validation for efficiency
        """
        answers_data = params["answers"]
        
        results = []
        approved_count = 0
        total_issues = 0
        
        for i, answer_data in enumerate(answers_data):
            validation_result = await self._validate_answer_tool({"answer": answer_data})
            results.append({
                "index": i,
                "question": answer_data["question"],
                "validation": validation_result
            })
            
            if validation_result["approved"]:
                approved_count += 1
            total_issues += len(validation_result["issues"])
        
        batch_summary = {
            "total_answers": len(answers_data),
            "approved_count": approved_count,
            "rejected_count": len(answers_data) - approved_count,
            "approval_rate": approved_count / len(answers_data) if answers_data else 0,
            "total_issues": total_issues,
            "avg_issues_per_answer": total_issues / len(answers_data) if answers_data else 0,
            "results": results
        }
        
        # Broadcast batch validation result
        await self.broadcast_event("batch_validation_completed", {
            "total_answers": len(answers_data),
            "approval_rate": batch_summary["approval_rate"],
            "total_issues": total_issues
        })
        
        return batch_summary
    
    async def _assess_answer_quality_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Assess detailed quality metrics
        
        Provides granular quality assessment beyond basic validation
        """
        answer_text = params["answer_text"]
        sources = params["sources"]
        expected_criteria = params.get("expected_criteria", ["length", "citations", "specificity", "completeness"])
        
        quality_scores = {}
        
        # Length assessment
        if "length" in expected_criteria:
            if len(answer_text) > 500:
                quality_scores["length"] = 1.0
            elif len(answer_text) > 200:
                quality_scores["length"] = 0.7
            elif len(answer_text) > 50:
                quality_scores["length"] = 0.4
            else:
                quality_scores["length"] = 0.1
        
        # Citation assessment
        if "citations" in expected_criteria:
            citation_patterns = ['page', 'source:', 'according to', 'document', 'pp.', 'from']
            citation_count = sum(1 for pattern in citation_patterns if pattern in answer_text.lower())
            quality_scores["citations"] = min(citation_count / 3, 1.0)  # Cap at 1.0
        
        # Specificity assessment
        if "specificity" in expected_criteria:
            specific_terms = ['specifically', 'precisely', 'exactly', 'detailed', 'particular']
            number_mentions = len(re.findall(r'\b\d+\b', answer_text))
            date_mentions = len(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b', answer_text))
            
            specificity_score = 0.0
            if any(term in answer_text.lower() for term in specific_terms):
                specificity_score += 0.3
            if number_mentions > 0:
                specificity_score += min(number_mentions / 5, 0.4)
            if date_mentions > 0:
                specificity_score += min(date_mentions / 3, 0.3)
            
            quality_scores["specificity"] = min(specificity_score, 1.0)
        
        # Completeness assessment
        if "completeness" in expected_criteria:
            completeness_indicators = ['comprehensive', 'complete', 'all', 'total', 'entire']
            incomplete_indicators = ['partial', 'some', 'limited', 'insufficient']
            
            completeness_score = 0.5  # Neutral baseline
            for indicator in completeness_indicators:
                if indicator in answer_text.lower():
                    completeness_score += 0.2
            for indicator in incomplete_indicators:
                if indicator in answer_text.lower():
                    completeness_score -= 0.2
            
            quality_scores["completeness"] = max(0.0, min(completeness_score, 1.0))
        
        # Overall quality score
        overall_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
        
        return {
            "quality_scores": quality_scores,
            "overall_quality": overall_quality,
            "quality_grade": self._get_quality_grade(overall_quality),
            "sources_count": len(sources),
            "answer_length": len(answer_text),
            "assessment_criteria": expected_criteria
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events from other agents"""
        if event_type == "research_completed":
            print(f"   📢 Research completed, ready for validation")
        
        elif event_type == "answer_validation_requested":
            print(f"   📢 Answer validation requested")

class MCPScoringAgent(MCPAgent):
    """
    MCP-enabled Scoring Agent
    
    Provides the same final scoring functionality as the original ScoringAgent
    but exposed as MCP tools with A2A communication capabilities.
    """
    
    def __init__(self, config: OptimizedConfig, message_bus):
        # Initialize LLM manager (same as original)
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
        
        # Initialize as MCP agent
        super().__init__("scoring_agent", config, message_bus)
        
        # Subscribe to relevant events
        self.subscribe_to_event("answer_validated")
        self.subscribe_to_event("batch_validation_completed")
        self.subscribe_to_event("evidence_collection_complete")
    
    def _setup_llm(self):
        """Setup LLM for scoring (same as original)"""
        self.llm, self.current_model = self.llm_manager.get_llm("scoring_agent")
    
    def _register_tools(self):
        """Register MCP tools for scoring"""
        
        # Main scoring tool
        self.register_tool(
            MCPToolSchema(
                name="score_topic",
                description="Score topic based on collected evidence and rubric",
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
                        "answers": {
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
                    "required": ["topic", "answers"]
                }
            ),
            self._score_topic_tool
        )
        
        # Evidence assessment tool
        self.register_tool(
            MCPToolSchema(
                name="assess_evidence_quality",
                description="Assess quality of collected evidence for scoring",
                input_schema={
                    "type": "object",
                    "properties": {
                        "answers": {"type": "array"},
                        "quality_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["confidence", "citations", "completeness", "consistency"]
                        }
                    },
                    "required": ["answers"]
                }
            ),
            self._assess_evidence_quality_tool
        )
        
        # Rubric analysis tool
        self.register_tool(
            MCPToolSchema(
                name="analyze_scoring_rubric",
                description="Analyze scoring rubric to understand requirements",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rubric": {"type": "object"},
                        "topic_context": {"type": "string"}
                    },
                    "required": ["rubric"]
                }
            ),
            self._analyze_scoring_rubric_tool
        )
    
    async def _score_topic_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Score topic based on evidence
        
        Same logic as original score_topic but as an MCP tool
        """
        topic_data = params["topic"]
        answers_data = params["answers"]
        
        # Convert to objects
        topic = TopicDefinition(
            topic_name=topic_data["topic_name"],
            goal=topic_data["goal"],
            guidance=topic_data["guidance"],
            scoring_rubric=topic_data["scoring_rubric"]
        )
        
        answers = [
            Answer(
                question=ans["question"],
                answer=ans["answer"],
                sources=ans["sources"],
                confidence=ans["confidence"],
                has_citations=ans["has_citations"]
            )
            for ans in answers_data
        ]
        
        if not self.llm:
            print(f"   ❌ No LLM available for scoring")
            result = {
                "score": 0,
                "justification": "No LLM available for scoring",
                "confidence": "low",
                "evidence_quality": "poor",
                "retrieval_method": self.config.retrieval_method
            }
            
            await self.broadcast_event("scoring_completed", {
                "topic_name": topic.topic_name,
                "score": 0,
                "success": False,
                "error": "No LLM available"
            })
            
            return result
        
        # Prepare evidence summary
        evidence_summary = self._prepare_evidence_summary(answers)
        print(f"   📊 Evidence summary prepared ({len(evidence_summary)} characters)")
        
        prompt = f"""
        You are scoring a corporate governance topic based on collected research evidence.
        
        TOPIC: {topic.topic_name}
        GOAL: {topic.goal}
        GUIDANCE: {topic.guidance}
        
        SCORING RUBRIC:
        {json.dumps(topic.scoring_rubric, indent=2)}
        
        RESEARCH EVIDENCE (collected using {self.config.retrieval_method} retrieval):
        {evidence_summary}
        
        Instructions:
        1. Evaluate the evidence against each scoring level in the rubric
        2. Assign a score (0, 1, or 2) based on which level best matches the evidence
        3. Provide detailed justification with specific references to the evidence
        4. Preserve all source citations from the evidence in your justification
        
        Respond in JSON format:
        {{
            "score": 0/1/2,
            "justification": "Detailed justification with source citations",
            "evidence_quality": "excellent/good/fair/poor",
            "key_findings": ["list of key findings that influenced the score"]
        }}
        
        Be objective and base your score strictly on the evidence provided.
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self.llm.invoke(prompt)
            
            # Parse response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate score
                score = result.get("score", 0)
                if score not in [0, 1, 2]:
                    score = 0
                    print(f"   ⚠️ Invalid score, defaulting to 0")
                
                confidence = self._assess_scoring_confidence(answers)
                
                print(f"   📊 Final Score: {score}/2")
                print(f"   🎯 Evidence Quality: {result.get('evidence_quality', 'fair')}")
                print(f"   💪 Scoring Confidence: {confidence}")
                
                final_result = {
                    "score": score,
                    "justification": result.get("justification", "No justification provided"),
                    "evidence_quality": result.get("evidence_quality", "fair"),
                    "key_findings": result.get("key_findings", []),
                    "confidence": confidence,
                    "retrieval_method": self.config.retrieval_method
                }
                
                # Broadcast scoring completed
                await self.broadcast_event("scoring_completed", {
                    "topic_name": topic.topic_name,
                    "score": score,
                    "evidence_quality": result.get('evidence_quality', 'fair'),
                    "confidence": confidence,
                    "success": True
                })
                
                return final_result
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            print(f"   ❌ Scoring failed: {str(e)}")
            
            result = {
                "score": 0,
                "justification": f"Scoring failed: {str(e)}",
                "confidence": "low",
                "evidence_quality": "poor",
                "retrieval_method": self.config.retrieval_method
            }
            
            await self.broadcast_event("scoring_completed", {
                "topic_name": topic.topic_name,
                "score": 0,
                "success": False,
                "error": str(e)
            })
            
            return result
    
    async def _assess_evidence_quality_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Assess quality of collected evidence
        
        Provides detailed assessment of evidence quality for transparency
        """
        answers_data = params["answers"]
        quality_criteria = params.get("quality_criteria", ["confidence", "citations", "completeness", "consistency"])
        
        if not answers_data:
            return {
                "overall_quality": "poor",
                "quality_scores": {},
                "evidence_count": 0,
                "recommendations": ["No evidence collected"]
            }
        
        # Convert to Answer objects
        answers = [
            Answer(
                question=ans["question"],
                answer=ans["answer"],
                sources=ans["sources"],
                confidence=ans["confidence"],
                has_citations=ans["has_citations"]
            )
            for ans in answers_data
        ]
        
        quality_scores = {}
        
        # Confidence assessment
        if "confidence" in quality_criteria:
            high_conf_count = sum(1 for ans in answers if ans.confidence == "high")
            confidence_score = high_conf_count / len(answers)
            quality_scores["confidence"] = confidence_score
        
        # Citations assessment
        if "citations" in quality_criteria:
            cited_count = sum(1 for ans in answers if ans.has_citations)
            citation_score = cited_count / len(answers)
            quality_scores["citations"] = citation_score
        
        # Completeness assessment
        if "completeness" in quality_criteria:
            total_length = sum(len(ans.answer) for ans in answers)
            avg_length = total_length / len(answers)
            completeness_score = min(avg_length / 500, 1.0)  # Normalize to 500 chars
            quality_scores["completeness"] = completeness_score
        
        # Consistency assessment
        if "consistency" in quality_criteria:
            all_sources = [source for ans in answers for source in ans.sources]
            unique_sources = set(all_sources)
            source_consistency = len(unique_sources) / len(all_sources) if all_sources else 0
            quality_scores["consistency"] = 1.0 - source_consistency  # Higher uniqueness = lower consistency
        
        # Overall quality
        overall_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
        
        # Generate recommendations
        recommendations = []
        if quality_scores.get("confidence", 0) < 0.5:
            recommendations.append("Improve answer confidence by using more reliable sources")
        if quality_scores.get("citations", 0) < 0.7:
            recommendations.append("Ensure all answers include proper source citations")
        if quality_scores.get("completeness", 0) < 0.6:
            recommendations.append("Provide more detailed and comprehensive answers")
        if quality_scores.get("consistency", 0) < 0.4:
            recommendations.append("Ensure consistency across multiple sources")
        
        # Determine overall quality grade
        if overall_score >= 0.8:
            overall_quality = "excellent"
        elif overall_score >= 0.6:
            overall_quality = "good"
        elif overall_score >= 0.4:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        return {
            "overall_quality": overall_quality,
            "overall_score": overall_score,
            "quality_scores": quality_scores,
            "evidence_count": len(answers),
            "recommendations": recommendations,
            "quality_breakdown": {
                "high_confidence_answers": sum(1 for ans in answers if ans.confidence == "high"),
                "cited_answers": sum(1 for ans in answers if ans.has_citations),
                "total_sources": len(set([source for ans in answers for source in ans.sources])),
                "avg_answer_length": sum(len(ans.answer) for ans in answers) / len(answers)
            }
        }
    
    async def _analyze_scoring_rubric_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Analyze scoring rubric to understand requirements
        
        Provides insights into rubric structure and requirements
        """
        rubric = params["rubric"]
        topic_context = params.get("topic_context", "")
        
        analysis = {
            "levels": list(rubric.keys()),
            "level_count": len(rubric),
            "complexity_analysis": {},
            "key_discriminators": [],
            "clarity_assessment": {}
        }
        
        # Analyze each level
        for level, criteria in rubric.items():
            if criteria and criteria.strip():
                word_count = len(criteria.split())
                analysis["complexity_analysis"][level] = {
                    "word_count": word_count,
                    "complexity": "high" if word_count > 20 else "medium" if word_count > 10 else "low"
                }
                
                # Extract key terms
                key_terms = self._extract_discriminating_terms(criteria)
                analysis["key_discriminators"].extend(key_terms)
        
        # Remove duplicates and analyze discriminators
        analysis["key_discriminators"] = list(set(analysis["key_discriminators"]))
        
        # Assess clarity
        for level, criteria in rubric.items():
            clarity_score = self._assess_criteria_clarity(criteria)
            analysis["clarity_assessment"][level] = clarity_score
        
        avg_clarity = sum(analysis["clarity_assessment"].values()) / len(analysis["clarity_assessment"])
        
        # Generate insights
        insights = []
        if len(analysis["levels"]) != 3:
            insights.append(f"Unusual number of scoring levels: {len(analysis['levels'])} (typically 3)")
        
        if avg_clarity < 0.6:
            insights.append("Some scoring criteria may be unclear or ambiguous")
        
        if len(analysis["key_discriminators"]) < 3:
            insights.append("Limited discriminating factors between scoring levels")
        
        return {
            "rubric_analysis": analysis,
            "avg_clarity_score": avg_clarity,
            "insights": insights,
            "scoring_recommendations": self._generate_scoring_recommendations(analysis)
        }
    
    def _prepare_evidence_summary(self, answers: List[Answer]) -> str:
        """Prepare summary of all research evidence (same as original)"""
        summary_parts = []
        
        for i, answer in enumerate(answers, 1):
            part = f"""
EVIDENCE {i}:
Question: {answer.question}
Answer: {answer.answer}
Sources: {', '.join(answer.sources)}
Confidence: {answer.confidence}
---
"""
            summary_parts.append(part)
        
        return "\n".join(summary_parts)
    
    def _assess_scoring_confidence(self, answers: List[Answer]) -> str:
        """Assess overall confidence in scoring (same as original)"""
        if not answers:
            return "low"
        
        high_confidence_count = sum(1 for ans in answers if ans.confidence == "high" and ans.has_citations)
        total_answers = len(answers)
        
        if high_confidence_count >= total_answers * 0.8:
            return "high"
        elif high_confidence_count >= total_answers * 0.5:
            return "medium"
        else:
            return "low"
    
    def _extract_discriminating_terms(self, criteria: str) -> List[str]:
        """Extract key terms that discriminate between scoring levels"""
        # Common governance discriminators
        discriminators = [
            "permanent", "independent", "lender", "representative", "disclosure",
            "complete", "detailed", "partial", "absent", "comprehensive",
            "adequate", "insufficient", "clear", "unclear", "specific"
        ]
        
        found_terms = []
        criteria_lower = criteria.lower()
        
        for term in discriminators:
            if term in criteria_lower:
                found_terms.append(term)
        
        return found_terms
    
    def _assess_criteria_clarity(self, criteria: str) -> float:
        """Assess clarity of scoring criteria"""
        if not criteria or not criteria.strip():
            return 0.0
        
        clarity_score = 0.5  # Base score
        
        # Positive indicators
        if any(word in criteria.lower() for word in ["specific", "clear", "detailed", "explicit"]):
            clarity_score += 0.2
        
        if len(criteria.split()) > 10:  # Sufficient detail
            clarity_score += 0.2
        
        if any(char in criteria for char in [".", ",", ";"]):  # Proper punctuation
            clarity_score += 0.1
        
        # Negative indicators
        if any(word in criteria.lower() for word in ["vague", "unclear", "ambiguous"]):
            clarity_score -= 0.2
        
        if len(criteria.split()) < 5:  # Too brief
            clarity_score -= 0.2
        
        return max(0.0, min(clarity_score, 1.0))
    
    def _generate_scoring_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for scoring based on rubric analysis"""
        recommendations = []
        
        if analysis["level_count"] != 3:
            recommendations.append("Consider using standard 3-level scoring rubric (0, 1, 2)")
        
        if len(analysis["key_discriminators"]) < 3:
            recommendations.append("Add more specific discriminating criteria between levels")
        
        # Check for clarity issues
        low_clarity_levels = [level for level, score in analysis["clarity_assessment"].items() if score < 0.6]
        if low_clarity_levels:
            recommendations.append(f"Improve clarity for levels: {', '.join(low_clarity_levels)}")
        
        return recommendations
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events from other agents"""
        if event_type == "answer_validated":
            approval_status = "approved" if data.get("approved", False) else "rejected"
            print(f"   📢 Answer validation completed: {approval_status}")
        
        elif event_type == "batch_validation_completed":
            approval_rate = data.get("approval_rate", 0)
            print(f"   📢 Batch validation completed: {approval_rate:.1%} approval rate")
        
        elif event_type == "evidence_collection_complete":
            print(f"   📢 Evidence collection complete, ready for scoring")

# Backward compatibility wrappers
class OutputGuardrailAgentWrapper:
    """Wrapper to maintain backward compatibility with original interface"""
    
    def __init__(self, mcp_agent: MCPOutputGuardrailAgent):
        self.mcp_agent = mcp_agent
    
    async def validate_answer(self, answer: Answer) -> Dict[str, Any]:
        """Original interface method - calls MCP tool internally"""
        return await self.mcp_agent.call_tool("validate_answer", {
            "answer": {
                "question": answer.question,
                "answer": answer.answer,
                "sources": answer.sources,
                "confidence": answer.confidence,
                "has_citations": answer.has_citations
            }
        })

class ScoringAgentWrapper:
    """Wrapper to maintain backward compatibility with original interface"""
    
    def __init__(self, mcp_agent: MCPScoringAgent):
        self.mcp_agent = mcp_agent
        self.current_model = mcp_agent.current_model
    
    async def score_topic(self, topic: TopicDefinition, answers: List[Answer]) -> Dict[str, Any]:
        """Original interface method - calls MCP tool internally"""
        return await self.mcp_agent.call_tool("score_topic", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            },
            "answers": [
                {
                    "question": ans.question,
                    "answer": ans.answer,
                    "sources": ans.sources,
                    "confidence": ans.confidence,
                    "has_citations": ans.has_citations
                }
                for ans in answers
            ]
        })

# Factory functions
def create_mcp_output_guardrail_agent(config: OptimizedConfig, message_bus) -> MCPOutputGuardrailAgent:
    """Factory function to create MCP Output Guardrail Agent"""
    return MCPOutputGuardrailAgent(config, message_bus)

def create_mcp_scoring_agent(config: OptimizedConfig, message_bus) -> MCPScoringAgent:
    """Factory function to create MCP Scoring Agent"""
    return MCPScoringAgent(config, message_bus)