"""
MCP Agentic Orchestrator - Multi-Agent Workflow Coordinator
===========================================================

Fully converted to use MCP tools and A2A communication while maintaining the same functionality
as the original OptimizedAgenticOrchestrator. This orchestrator coordinates all MCP agents to 
perform complete corporate governance evaluations using standardized protocols.

Key Changes:
- All agent interactions use MCP tools via A2A messages
- Maintains same evaluate_topic() interface for backward compatibility
- Enhanced observability with A2A message tracing
- Distributed agent architecture with centralized coordination
- Same performance optimizations with protocol benefits
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import base framework
from mcp_a2a_base import (
    MCPAgentManager, A2AMessageBus, get_agent_manager, 
    print_mcp_action, print_a2a_message, Colors
)

# Import MCP agents
from mcp_input_agent import create_mcp_input_guardrail_agent, InputGuardrailAgentWrapper
from mcp_question_agent import create_mcp_question_agent, QuestionAgentWrapper  
from mcp_research_agent import create_mcp_research_agent, OptimizedResearchAgentWrapper
from mcp_output_scoring_agents import (
    create_mcp_output_guardrail_agent, create_mcp_scoring_agent,
    OutputGuardrailAgentWrapper, ScoringAgentWrapper
)

# Import original dependencies
from main import (
    OptimizedConfig, TopicDefinition, Question, Answer, 
    print_section, Colors as OrigColors
)

logger = logging.getLogger(__name__)

class MCPAgenticOrchestrator:
    """
    MCP-enabled Agentic Orchestrator
    
    Coordinates all MCP agents to perform complete corporate governance evaluations
    using MCP tools and A2A communication while maintaining the same interfaces
    and functionality as the original system.
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
        # Create centralized message bus and agent manager
        self.agent_manager = get_agent_manager()
        self.message_bus = self.agent_manager.message_bus
        
        # Initialize all MCP agents
        self._initialize_mcp_agents()
        
        # Create backward compatibility wrappers
        self._create_compatibility_wrappers()
        
        # State tracking (same as original)
        self.current_topic = None
        self.answers = []
        self.max_iterations = 3
        
        # Display configuration
        self._display_agent_configuration()
        
        print_section("MCP AGENTIC ORCHESTRATOR READY", 
                     f"Retrieval: {config.retrieval_method}, Max Iterations: {self.max_iterations}\n" +
                     f"✅ All agents initialized with MCP tools\n" +
                     f"✅ A2A communication bus active\n" +
                     f"⚡ Ready for FAST distributed queries", OrigColors.OKGREEN)
    
    def _initialize_mcp_agents(self):
        """Initialize all MCP agents and register with agent manager"""
        
        print_section("INITIALIZING MCP AGENTS", color=OrigColors.HEADER)
        
        # Create MCP agents
        self.input_guardrail_agent = create_mcp_input_guardrail_agent(self.config, self.message_bus)
        self.question_agent = create_mcp_question_agent(self.config, self.message_bus)
        self.research_agent = create_mcp_research_agent(self.config, self.message_bus)
        self.output_guardrail_agent = create_mcp_output_guardrail_agent(self.config, self.message_bus)
        self.scoring_agent = create_mcp_scoring_agent(self.config, self.message_bus)
        
        # Register all agents with the manager
        self.agent_manager.register_agent(self.input_guardrail_agent)
        self.agent_manager.register_agent(self.question_agent)
        self.agent_manager.register_agent(self.research_agent)
        self.agent_manager.register_agent(self.output_guardrail_agent)
        self.agent_manager.register_agent(self.scoring_agent)
        
        print(f"   ✅ Registered {len(self.agent_manager.agents)} MCP agents")
        print(f"   📡 A2A message bus initialized")
        print(f"   🛠️ Total MCP tools available: {self._count_total_tools()}")
    
    def _create_compatibility_wrappers(self):
        """Create backward compatibility wrappers for original interfaces"""
        
        # These wrappers allow existing code to work unchanged
        self.input_guardrail = InputGuardrailAgentWrapper(self.input_guardrail_agent)
        self.question_agent_wrapper = QuestionAgentWrapper(self.question_agent)
        self.research_agent_wrapper = OptimizedResearchAgentWrapper(self.research_agent)
        self.output_guardrail = OutputGuardrailAgentWrapper(self.output_guardrail_agent)
        self.scoring_agent_wrapper = ScoringAgentWrapper(self.scoring_agent)
        
        print(f"   🔄 Backward compatibility wrappers created")
    
    async def evaluate_topic(self, topic: TopicDefinition) -> Dict[str, Any]:
        """
        Main evaluation workflow using MCP agents and A2A communication
        
        Same interface as original but uses MCP tools internally
        """
        
        print_section("STARTING MCP AGENTIC EVALUATION", 
                     f"Topic: {topic.topic_name}\nMethod: {self.config.retrieval_method}", 
                     OrigColors.HEADER)
        
        total_start_time = time.time()
        
        # Broadcast evaluation started event
        await self.agent_manager.broadcast_system_event("evaluation_started", {
            "topic_name": topic.topic_name,
            "retrieval_method": self.config.retrieval_method,
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 1: Input Validation via MCP
        print_section("STEP 1: MCP INPUT VALIDATION")
        input_validation = await self._validate_topic_via_mcp(topic)
        if not input_validation.get("valid", True):
            print(f"   ❌ Validation failed")
            return {
                "success": False,
                "error": "Invalid topic definition",
                "issues": input_validation.get("issues", []),
                "suggestions": input_validation.get("suggestions", [])
            }
        
        # Initialize state
        self.current_topic = topic
        self.answers = []
        iteration = 0
        
        # Step 2: Generate initial question via MCP
        print_section("STEP 2: MCP QUESTION GENERATION")
        question_start = time.time()
        current_question = await self._generate_initial_question_via_mcp(topic)
        question_time = time.time() - question_start
        print(f"   ⏱️ Question generation: {question_time:.3f}s")
        
        # Step 3: Iterative research with MCP agents
        print_section("STEP 3: MCP ITERATIVE RESEARCH")
        
        total_research_time = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print_section(f"MCP RESEARCH ITERATION {iteration}", 
                         f"Question: {current_question['text']}", 
                         OrigColors.WARNING)
            
            # FAST research using MCP research agent
            research_start = time.time()
            answer = await self._research_question_via_mcp(current_question["text"])
            research_time = time.time() - research_start
            total_research_time += research_time
            
            print(f"   ⚡ MCP research iteration completed in {research_time:.3f}s")
            
            # Validate the answer via MCP
            validation = await self._validate_answer_via_mcp(answer)
            
            if validation["approved"]:
                self.answers.append(answer)
                print(f"   ✅ Answer approved via MCP validation")
            else:
                print(f"   ⚠️ Answer has issues but still added to evidence pool")
                self.answers.append(answer)
            
            print(f"   📊 Current evidence count: {len(self.answers)}")
            
            # Determine if we need more information via MCP
            follow_up_question = await self._generate_follow_up_question_via_mcp(topic, self.answers)
            
            if follow_up_question is None:
                print(f"   ✅ MCP agents determined no more questions needed")
                break
            else:
                print(f"   ➡️ MCP follow-up question generated")
                current_question = follow_up_question
        
        # Step 4: Final scoring via MCP
        print_section("STEP 4: MCP FINAL SCORING")
        scoring_start = time.time()
        scoring_result = await self._score_topic_via_mcp(topic, self.answers)
        scoring_time = time.time() - scoring_start
        print(f"   ⏱️ MCP scoring completed in {scoring_time:.3f}s")
        
        # Step 5: Compile final result
        print_section("STEP 5: COMPILING RESULTS")
        
        total_time = time.time() - total_start_time
        
        result = await self._compile_final_result(
            topic, self.answers, scoring_result, iteration,
            total_time, total_research_time, question_time, scoring_time
        )
        
        # Broadcast evaluation completed event
        await self.agent_manager.broadcast_system_event("evaluation_completed", {
            "topic_name": topic.topic_name,
            "success": result["success"],
            "final_score": result.get("scoring", {}).get("score", 0),
            "total_time": total_time,
            "timestamp": datetime.now().isoformat()
        })
        
        print_section("MCP AGENTIC EVALUATION COMPLETED", 
                     f"Final Score: {scoring_result.get('score', 'N/A')}/2\n" +
                     f"Total Time: {total_time:.3f}s (Research: {total_research_time:.3f}s)\n" +
                     f"Avg Research/Iteration: {result['performance_metrics']['avg_research_per_iteration']:.3f}s\n" +
                     f"Sources Used: {result['research_summary']['total_sources']}\n" +
                     f"Method: MCP {self.config.retrieval_method} (Distributed)",
                     OrigColors.OKGREEN)
        
        return result
    
    async def _validate_topic_via_mcp(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Validate topic using MCP input guardrail agent"""
        
        return await self.input_guardrail_agent.call_tool("validate_topic_definition", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }
        })
    
    async def _generate_initial_question_via_mcp(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Generate initial question using MCP question agent"""
        
        return await self.question_agent.call_tool("generate_initial_question", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }
        })
    
    async def _research_question_via_mcp(self, question: str) -> Dict[str, Any]:
        """Research question using MCP research agent"""
        
        return await self.research_agent.call_tool("research_question", {
            "question": question
        })
    
    async def _validate_answer_via_mcp(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """Validate answer using MCP output guardrail agent"""
        
        return await self.output_guardrail_agent.call_tool("validate_answer", {
            "answer": {
                "question": answer["question"],
                "answer": answer["answer"],
                "sources": answer["sources"],
                "confidence": answer["confidence"],
                "has_citations": answer["has_citations"]
            }
        })
    
    async def _generate_follow_up_question_via_mcp(self, topic: TopicDefinition, 
                                                   answers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate follow-up question using MCP question agent"""
        
        result = await self.question_agent.call_tool("generate_follow_up_question", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            },
            "existing_answers": [
                {
                    "question": ans["question"],
                    "answer": ans["answer"],
                    "sources": ans["sources"],
                    "confidence": ans["confidence"],
                    "has_citations": ans["has_citations"]
                }
                for ans in answers
            ]
        })
        
        return result  # None if no follow-up needed
    
    async def _score_topic_via_mcp(self, topic: TopicDefinition, 
                                   answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score topic using MCP scoring agent"""
        
        return await self.scoring_agent.call_tool("score_topic", {
            "topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            },
            "answers": [
                {
                    "question": ans["question"],
                    "answer": ans["answer"],
                    "sources": ans["sources"],
                    "confidence": ans["confidence"],
                    "has_citations": ans["has_citations"]
                }
                for ans in answers
            ]
        })
    
    async def _compile_final_result(self, topic: TopicDefinition, answers: List[Dict[str, Any]], 
                                   scoring_result: Dict[str, Any], iteration: int,
                                   total_time: float, total_research_time: float, 
                                   question_time: float, scoring_time: float) -> Dict[str, Any]:
        """Compile comprehensive final result with MCP-specific metrics"""
        
        # Get agent model information
        agent_models = {}
        for agent_id, agent in self.agent_manager.agents.items():
            if hasattr(agent, 'current_model'):
                agent_models[agent_id] = agent.current_model
            else:
                agent_models[agent_id] = "No LLM" if agent_id == "output_guardrail" else "Unknown"
        
        # Calculate unique sources
        all_sources = [source for ans in answers for source in ans["sources"]]
        unique_sources = len(set(all_sources))
        
        # Validate answers for approval count
        validation_results = []
        for answer in answers:
            validation = await self.output_guardrail_agent.call_tool("validate_answer", {
                "answer": answer
            })
            validation_results.append(validation["approved"])
        
        approved_count = sum(validation_results)
        
        result = {
            "success": True,
            "topic": {
                "name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "rubric": topic.scoring_rubric
            },
            "research_summary": {
                "iterations": iteration,
                "questions_asked": len(answers),
                "answers_approved": approved_count,
                "retrieval_method": self.config.retrieval_method,
                "total_sources": unique_sources,
                "pdf_slices_used": self.config.use_pdf_slices,
                "optimization_enabled": True,
                "mcp_enabled": True,  # New field indicating MCP usage
                "a2a_enabled": True,  # New field indicating A2A usage
                "agent_models": agent_models,  # Model information per agent
                "message_count": len(self.message_bus.message_history)  # A2A message count
            },
            "performance_metrics": {
                "total_time": total_time,
                "research_time": total_research_time,
                "question_time": question_time,
                "scoring_time": scoring_time,
                "avg_research_per_iteration": total_research_time / iteration if iteration > 0 else 0,
                "mcp_overhead": self._calculate_mcp_overhead(),  # Protocol overhead
                "a2a_message_latency": self._calculate_avg_message_latency()  # Communication latency
            },
            "evidence": answers,  # Already in correct format
            "scoring": scoring_result,
            "mcp_metrics": await self._get_mcp_metrics(),  # MCP-specific metrics
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _calculate_mcp_overhead(self) -> float:
        """Calculate MCP protocol overhead using recorded message timings"""
        latencies = []
        for msg in self.message_bus.message_history:
            if msg.message_type == "request" and hasattr(msg, "response_timestamp"):
                try:
                    start = datetime.fromisoformat(msg.timestamp)
                    end = datetime.fromisoformat(msg.response_timestamp)
                    latencies.append((end - start).total_seconds())
                except Exception:
                    continue

        if not latencies:
            return 0.0

        # Average overhead per request/response pair
        return sum(latencies) / len(latencies)
    
    def _calculate_avg_message_latency(self) -> float:
        """Calculate average A2A message latency from history"""

        latencies = []
        for msg in self.message_bus.message_history:
            if msg.message_type == "request" and hasattr(msg, "response_timestamp"):
                try:
                    start = datetime.fromisoformat(msg.timestamp)
                    end = datetime.fromisoformat(msg.response_timestamp)
                    latencies.append((end - start).total_seconds())
                except Exception:
                    continue

        if not latencies:
            return 0.0

        return sum(latencies) / len(latencies)
    
    async def _get_mcp_metrics(self) -> Dict[str, Any]:
        """Get comprehensive MCP-specific metrics"""
        
        # Get tool usage statistics
        tool_usage = {}
        for agent_id, agent in self.agent_manager.agents.items():
            tool_schemas = agent.get_tool_schemas()
            tool_usage[agent_id] = {
                "available_tools": len(tool_schemas),
                "tool_names": [tool["name"] for tool in tool_schemas]
            }
        
        # Get A2A message statistics
        message_types = {}
        for message in self.message_bus.message_history:
            msg_type = message.message_type
            if msg_type not in message_types:
                message_types[msg_type] = 0
            message_types[msg_type] += 1
        
        # Get agent capabilities overview
        agent_capabilities = self.agent_manager.get_agent_capabilities()
        
        return {
            "total_agents": len(self.agent_manager.agents),
            "total_tools": sum(usage["available_tools"] for usage in tool_usage.values()),
            "tool_usage_by_agent": tool_usage,
            "message_statistics": {
                "total_messages": len(self.message_bus.message_history),
                "message_types": message_types,
                "avg_messages_per_agent": len(self.message_bus.message_history) / len(self.agent_manager.agents)
            },
            "agent_capabilities": agent_capabilities,
            "protocol_version": "MCP-1.0-A2A-1.0"
        }
    
    def _count_total_tools(self) -> int:
        """Count total number of MCP tools across all agents"""
        total = 0
        for agent in self.agent_manager.agents.values():
            total += len(agent.tools)
        return total
    
    def _display_agent_configuration(self):
        """Display current MCP agent configuration"""
        print_section("MCP AGENT CONFIGURATION", color=OrigColors.HEADER)
        
        # Get actual models being used
        agent_info = {}
        for agent_id, agent in self.agent_manager.agents.items():
            model = getattr(agent, 'current_model', 'No LLM' if agent_id == 'output_guardrail' else 'Unknown')
            tool_count = len(agent.tools)
            agent_info[agent_id] = {"model": model, "tools": tool_count}
        
        print(f"   🛡️  Input Guardrail Agent:   {agent_info['input_guardrail']['model']} ({agent_info['input_guardrail']['tools']} tools)")
        print(f"   ❓  Question Agent:          {agent_info['question_agent']['model']} ({agent_info['question_agent']['tools']} tools)")
        print(f"   🔍  Research Agent:          {agent_info['research_agent']['model']} ({agent_info['research_agent']['tools']} tools)")
        print(f"   🏛️  Output Guardrail Agent:  {agent_info['output_guardrail']['model']} ({agent_info['output_guardrail']['tools']} tools)")
        print(f"   📊  Scoring Agent:           {agent_info['scoring_agent']['model']} ({agent_info['scoring_agent']['tools']} tools)")
        
        # Show model distribution
        models_used = [info["model"] for info in agent_info.values() if info["model"] not in ["No LLM", "Unknown"]]
        gemini_count = len([m for m in models_used if m.startswith('gemini')])
        ollama_count = len([m for m in models_used if not m.startswith('gemini')])
        total_tools = sum(info["tools"] for info in agent_info.values())
        
        print(f"\n   📈 System Statistics:")
        print(f"      🛠️  Total MCP tools: {total_tools}")
        print(f"      📡  A2A message bus: Active")
        if gemini_count > 0:
            print(f"      🌩️  Gemini models: {gemini_count}/4 LLM agents")
        if ollama_count > 0:
            print(f"      🏠  Local models:  {ollama_count}/4 LLM agents")
        
        print()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring"""
        
        return {
            "orchestrator_status": "active",
            "agent_count": len(self.agent_manager.agents),
            "message_bus_status": "active" if self.message_bus else "inactive",
            "total_tools": self._count_total_tools(),
            "message_history_size": len(self.message_bus.message_history),
            "current_topic": self.current_topic.topic_name if self.current_topic else None,
            "evidence_collected": len(self.answers),
            "configuration": {
                "retrieval_method": self.config.retrieval_method,
                "max_iterations": self.max_iterations,
                "use_pdf_slices": self.config.use_pdf_slices,
                "auto_fallback": self.config.auto_fallback_to_direct
            },
            "capabilities": self.agent_manager.get_agent_capabilities()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator and all agents"""
        print_section("SHUTTING DOWN MCP ORCHESTRATOR", color=OrigColors.WARNING)
        
        try:
            # Broadcast shutdown event
            await self.agent_manager.broadcast_system_event("system_shutdown", {
                "timestamp": datetime.now().isoformat(),
                "reason": "Graceful shutdown"
            })
            
            # Shutdown all agents
            await self.agent_manager.shutdown_all_agents()
            
            print(f"   ✅ All agents shutdown successfully")
            print(f"   📡 A2A message bus stopped")
            print(f"   💾 Message history: {len(self.message_bus.message_history)} messages")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            print(f"   ❌ Shutdown error: {e}")

# Factory function for creating MCP orchestrator
def create_mcp_orchestrator(config: OptimizedConfig) -> MCPAgenticOrchestrator:
    """Factory function to create MCP orchestrator"""
    return MCPAgenticOrchestrator(config)

# Async wrapper for backward compatibility
class MCPAgenticOrchestratorSyncWrapper:
    """
    Synchronous wrapper for backward compatibility
    
    Allows existing synchronous code to work with the async MCP orchestrator
    """
    
    def __init__(self, config: OptimizedConfig):
        self.async_orchestrator = MCPAgenticOrchestrator(config)
        self.loop = None
    
    def _get_or_create_loop(self):
        """Get or create event loop for async operations"""
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop
    
    def evaluate_topic(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Synchronous wrapper for evaluate_topic"""
        loop = self._get_or_create_loop()
        
        if loop.is_running():
            # If loop is already running, we need to use a different approach
            # This typically happens in Jupyter notebooks or some web frameworks
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.async_orchestrator.evaluate_topic(topic))
        else:
            return loop.run_until_complete(self.async_orchestrator.evaluate_topic(topic))
    
    def get_system_status(self) -> Dict[str, Any]:
        """Synchronous wrapper for get_system_status"""
        loop = self._get_or_create_loop()
        return loop.run_until_complete(self.async_orchestrator.get_system_status())
    
    def shutdown(self):
        """Synchronous wrapper for shutdown"""
        loop = self._get_or_create_loop()
        loop.run_until_complete(self.async_orchestrator.shutdown())
    
    # Expose attributes for compatibility
    @property
    def max_iterations(self):
        return self.async_orchestrator.max_iterations
    
    @max_iterations.setter
    def max_iterations(self, value):
        self.async_orchestrator.max_iterations = value
    
    @property
    def config(self):
        return self.async_orchestrator.config

# For complete backward compatibility, alias the sync wrapper
OptimizedAgenticOrchestrator = MCPAgenticOrchestratorSyncWrapper

# Example usage and testing functions
async def test_mcp_orchestrator(company: str = "PAYTM"):
    """Test the MCP orchestrator with a sample topic"""
    
    from main import create_sample_topic, OptimizedConfig
    
    print_section("TESTING MCP AGENTIC ORCHESTRATOR", 
                 f"Company: {company}", OrigColors.HEADER)
    
    # Create config and orchestrator
    config = OptimizedConfig(company)
    orchestrator = MCPAgenticOrchestrator(config)
    
    # Create sample topic
    topic = create_sample_topic()
    
    try:
        # Run evaluation
        result = await orchestrator.evaluate_topic(topic)
        
        if result["success"]:
            print_section("MCP EVALUATION SUCCESS", 
                         f"Score: {result['scoring']['score']}/2\n" +
                         f"Evidence Quality: {result['scoring']['evidence_quality']}\n" +
                         f"Total Time: {result['performance_metrics']['total_time']:.3f}s\n" +
                         f"MCP Tools Used: {result['mcp_metrics']['total_tools']}\n" +
                         f"A2A Messages: {result['mcp_metrics']['message_statistics']['total_messages']}",
                         OrigColors.OKGREEN)
            
            return result
        else:
            print_section("MCP EVALUATION FAILED", 
                         result.get('error', 'Unknown error'), OrigColors.FAIL)
            return result
            
    except Exception as e:
        print_section("MCP TEST ERROR", str(e), OrigColors.FAIL)
        return {"success": False, "error": str(e)}
    
    finally:
        # Cleanup
        await orchestrator.shutdown()

def test_mcp_orchestrator_sync(company: str = "PAYTM"):
    """Synchronous test function for backward compatibility"""
    
    from main import create_sample_topic, OptimizedConfig
    
    print_section("TESTING MCP ORCHESTRATOR (SYNC)", 
                 f"Company: {company}", OrigColors.HEADER)
    
    # Create config and sync orchestrator
    config = OptimizedConfig(company)
    orchestrator = OptimizedAgenticOrchestrator(config)  # Sync wrapper
    
    # Create sample topic
    topic = create_sample_topic()
    
    try:
        # Run evaluation (synchronously)
        result = orchestrator.evaluate_topic(topic)
        
        if result["success"]:
            print_section("SYNC MCP EVALUATION SUCCESS", 
                         f"Score: {result['scoring']['score']}/2\n" +
                         f"Total Time: {result['performance_metrics']['total_time']:.3f}s",
                         OrigColors.OKGREEN)
        else:
            print_section("SYNC MCP EVALUATION FAILED", 
                         result.get('error', 'Unknown error'), OrigColors.FAIL)
        
        return result
        
    finally:
        # Cleanup
        orchestrator.shutdown()

if __name__ == "__main__":
    # Test MCP orchestrator
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        # Run async test
        asyncio.run(test_mcp_orchestrator())
    else:
        # Run sync test (default)
        test_mcp_orchestrator_sync()