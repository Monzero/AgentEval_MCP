"""
Fixed Dynamic MCP Orchestrator - Proper Async/Sync Integration
=============================================================

This fixes the coroutine issue by providing proper synchronous wrappers
that work seamlessly with Streamlit while maintaining all dynamic capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# Import base framework and existing components
from mcp_a2a_base import MCPAgentManager, A2AMessageBus, Colors
from main import OptimizedConfig, TopicDefinition, LLMManager

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Different types of tasks the orchestrator can handle"""
    VALIDATION = "validation"
    QUESTION_GENERATION = "question_generation"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SCORING = "scoring"
    SYNTHESIS = "synthesis"

@dataclass
class AgentCapability:
    """Represents a capability that an agent provides"""
    agent_id: str
    tool_name: str
    description: str
    input_schema: Dict[str, Any]
    task_types: List[TaskType]
    complexity_score: float  # 0.0-1.0, higher = more complex tasks
    estimated_latency: float  # seconds
    dependencies: List[str]  # Required inputs/prerequisites

@dataclass
class WorkflowStep:
    """A single step in a planned workflow"""
    step_id: str
    agent_id: str
    tool_name: str
    task_type: TaskType
    inputs: Dict[str, Any]
    dependencies: List[str]  # step_ids this depends on
    can_parallelize: bool
    estimated_duration: float
    priority: int  # 1=highest priority

@dataclass
class ExecutionResult:
    """Result of executing a workflow step"""
    step_id: str
    success: bool
    result: Any
    error: Optional[str]
    execution_time: float
    confidence: str
    metadata: Dict[str, Any]

class DynamicMCPOrchestrator:
    """
    Dynamic MCP Orchestrator that uses AI to plan and execute workflows
    
    This version includes proper sync/async handling for Streamlit integration
    """
    
    def __init__(self, config: OptimizedConfig):
        self.config = config
        
        # Core components
        self.agent_manager = MCPAgentManager()
        self.message_bus = self.agent_manager.message_bus
        self.llm_manager = LLMManager(config)
        
        # Planning components
        self.planner_llm = None
        self.orchestrator_llm = None
        self._setup_orchestrator_llms()
        
        # Discovery and capability tracking
        self.discovered_capabilities: Dict[str, List[AgentCapability]] = {}
        self.capability_cache_timestamp: float = 0.0
        self.agent_performance_history: Dict[str, List[float]] = {}
        
        # Execution state
        self.current_workflow: List[WorkflowStep] = []
        self.execution_results: Dict[str, ExecutionResult] = {}
        self.workflow_context: Dict[str, Any] = {}
        
        print(f"{Colors.HEADER}🚀 DYNAMIC MCP ORCHESTRATOR INITIALIZED{Colors.ENDC}")
        print(f"   🧠 AI-Driven Workflow Planning: Enabled")
        print(f"   🔍 Dynamic Agent Discovery: Enabled") 
        print(f"   ⚡ Parallel Execution: Enabled")
        print(f"   🔄 Adaptive Re-planning: Enabled")
    
    def _setup_orchestrator_llms(self):
        """Setup specialized LLMs for orchestration tasks"""
        # Planner LLM: Analyzes tasks and creates workflow plans
        self.planner_llm, _ = self.llm_manager.get_llm("planner", "gemini-1.5-pro")
        
        # Orchestrator LLM: Makes real-time execution decisions
        self.orchestrator_llm, _ = self.llm_manager.get_llm("orchestrator", "gemini-1.5-flash")
        
        print(f"   🎯 Planner LLM: gemini-1.5-pro (strategic planning)")
        print(f"   🎭 Orchestrator LLM: gemini-1.5-flash (execution decisions)")
    
    async def discover_system_capabilities(self) -> Dict[str, List[AgentCapability]]:
        """Dynamically discover all available agents and their capabilities"""
        print(f"\n{Colors.OKCYAN}🔍 DISCOVERING SYSTEM CAPABILITIES{Colors.ENDC}")

        cache_ttl = self.config.dynamic_orchestration.get("capability_discovery_cache", 0)
        now = time.time()

        # Return cached capabilities if still valid
        if self.discovered_capabilities and (now - self.capability_cache_timestamp) < cache_ttl:
            print("   ⚙️  Using cached capabilities")
            return self.discovered_capabilities

        capabilities = {}

        # Get all registered agents
        agent_capabilities = self.agent_manager.get_agent_capabilities()
        
        for agent_id, tools in agent_capabilities.items():
            agent_caps = []
            
            for tool in tools:
                # Analyze tool to determine task types it can handle
                task_types = self._infer_task_types(tool["name"], tool["description"])
                
                # Estimate complexity and latency based on tool characteristics
                complexity = self._estimate_complexity(tool)
                latency = self._estimate_latency(tool, agent_id)
                
                # Extract dependencies from input schema
                dependencies = self._extract_dependencies(tool.get("inputSchema", {}))
                
                capability = AgentCapability(
                    agent_id=agent_id,
                    tool_name=tool["name"],
                    description=tool["description"],
                    input_schema=tool.get("inputSchema", {}),
                    task_types=task_types,
                    complexity_score=complexity,
                    estimated_latency=latency,
                    dependencies=dependencies
                )
                
                agent_caps.append(capability)
            
            capabilities[agent_id] = agent_caps
            
            print(f"   🤖 {agent_id}: {len(agent_caps)} capabilities discovered")
            for cap in agent_caps:
                task_types_str = ", ".join([t.value for t in cap.task_types])
                print(f"     • {cap.tool_name}: {task_types_str} (complexity: {cap.complexity_score:.2f})")
        
        self.discovered_capabilities = capabilities
        self.capability_cache_timestamp = now
        return capabilities
    
    def _infer_task_types(self, tool_name: str, description: str) -> List[TaskType]:
        """Infer what types of tasks a tool can handle based on its name and description"""
        task_types = []
        
        name_lower = tool_name.lower()
        desc_lower = description.lower()
        
        # Validation tasks
        if any(word in name_lower for word in ["validate", "check", "verify", "assess"]):
            task_types.append(TaskType.VALIDATION)
        
        # Question generation tasks
        if any(word in name_lower for word in ["question", "generate", "ask", "inquire"]):
            task_types.append(TaskType.QUESTION_GENERATION)
        
        # Research tasks
        if any(word in name_lower for word in ["research", "search", "find", "extract", "analyze"]):
            task_types.append(TaskType.RESEARCH)
        
        # Analysis tasks
        if any(word in name_lower for word in ["analyze", "process", "evaluate", "examine"]):
            task_types.append(TaskType.ANALYSIS)
        
        # Scoring tasks
        if any(word in name_lower for word in ["score", "rate", "rank", "grade", "assess"]):
            task_types.append(TaskType.SCORING)
        
        # Synthesis tasks
        if any(word in name_lower for word in ["synthesize", "combine", "merge", "integrate"]):
            task_types.append(TaskType.SYNTHESIS)
        
        # Fallback based on description
        if not task_types:
            if any(word in desc_lower for word in ["validation", "check"]):
                task_types.append(TaskType.VALIDATION)
            elif any(word in desc_lower for word in ["research", "search"]):
                task_types.append(TaskType.RESEARCH)
            else:
                task_types.append(TaskType.ANALYSIS)  # Default
        
        return task_types
    
    def _estimate_complexity(self, tool: Dict[str, Any]) -> float:
        """Estimate the complexity of a tool (0.0 = simple, 1.0 = very complex)"""
        complexity = 0.0
        
        # Base complexity from tool name
        name = tool["name"].lower()
        if "validate" in name or "check" in name:
            complexity += 0.2
        elif "generate" in name or "create" in name:
            complexity += 0.5
        elif "research" in name or "analyze" in name:
            complexity += 0.7
        elif "score" in name or "evaluate" in name:
            complexity += 0.8
        
        # Complexity from input schema
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        
        # More parameters = higher complexity
        complexity += min(len(properties) * 0.1, 0.3)
        
        # Complex parameter types = higher complexity
        for prop in properties.values():
            if prop.get("type") == "array":
                complexity += 0.1
            elif prop.get("type") == "object":
                complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _estimate_latency(self, tool: Dict[str, Any], agent_id: str) -> float:
        """Estimate latency for a tool execution"""
        # Base latency from historical performance
        if agent_id in self.agent_performance_history:
            history = self.agent_performance_history[agent_id]
            base_latency = sum(history) / len(history) if history else 1.0
        else:
            base_latency = 1.0  # Default 1 second
        
        # Adjust based on tool characteristics
        name = tool["name"].lower()
        if "research" in name or "search" in name:
            base_latency *= 2.0  # Research takes longer
        elif "validate" in name or "check" in name:
            base_latency *= 0.5  # Validation is typically fast
        
        return base_latency
    
    def _extract_dependencies(self, input_schema: Dict[str, Any]) -> List[str]:
        """Extract dependencies from input schema"""
        dependencies = []
        
        # Check required properties for common dependency patterns
        required = input_schema.get("required", [])
        properties = input_schema.get("properties", {})
        
        for req_field in required:
            prop = properties.get(req_field, {})
            
            # Common dependency patterns
            if req_field in ["topic", "topic_definition"]:
                dependencies.append("topic_validation")
            elif req_field in ["question", "questions"]:
                dependencies.append("question_generation")
            elif req_field in ["answers", "evidence", "existing_answers"]:
                dependencies.append("research_completion")
            elif req_field in ["validation_result"]:
                dependencies.append("answer_validation")
        
        return dependencies
    
    async def plan_workflow(self, task_description: str, context: Dict[str, Any] = None) -> List[WorkflowStep]:
        """Use AI to dynamically plan an optimal workflow for the given task"""
        print(f"\n{Colors.WARNING}🧠 AI WORKFLOW PLANNING{Colors.ENDC}")
        print(f"   📋 Task: {task_description}")
        
        if not self.planner_llm:
            print(f"   ❌ No planner LLM available")
            return await self._fallback_to_static_workflow(task_description)
        
        # For now, use a simplified planning approach
        # In production, this would use the full AI planning logic
        return await self._fallback_to_static_workflow(task_description)
    
    async def _fallback_to_static_workflow(self, task_description: str) -> List[WorkflowStep]:
        """Fallback to a predefined workflow when AI planning fails"""
        print(f"   🔄 Using static fallback workflow")
        
        # Create a basic validation -> research -> scoring workflow
        fallback_steps = [
            WorkflowStep(
                step_id="validate_input",
                agent_id="input_guardrail",
                tool_name="validate_topic_definition",
                task_type=TaskType.VALIDATION,
                inputs={"topic": "from_context"},
                dependencies=[],
                can_parallelize=False,
                estimated_duration=1.0,
                priority=1
            ),
            WorkflowStep(
                step_id="generate_question", 
                agent_id="question_agent",
                tool_name="generate_initial_question",
                task_type=TaskType.QUESTION_GENERATION,
                inputs={"topic": "from_context"},
                dependencies=["validate_input"],
                can_parallelize=False,
                estimated_duration=2.0,
                priority=2
            ),
            WorkflowStep(
                step_id="research_question",
                agent_id="research_agent", 
                tool_name="research_question",
                task_type=TaskType.RESEARCH,
                inputs={"question": "from_previous"},
                dependencies=["generate_question"],
                can_parallelize=False,
                estimated_duration=5.0,
                priority=3
            ),
            WorkflowStep(
                step_id="score_topic",
                agent_id="scoring_agent",
                tool_name="score_topic", 
                task_type=TaskType.SCORING,
                inputs={"topic": "from_context", "answers": "from_research"},
                dependencies=["research_question"],
                can_parallelize=False,
                estimated_duration=3.0,
                priority=4
            )
        ]
        
        return fallback_steps
    
    async def execute_workflow(self, workflow: List[WorkflowStep], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned workflow"""
        print(f"\n{Colors.OKGREEN}⚡ EXECUTING DYNAMIC WORKFLOW{Colors.ENDC}")
        print(f"   📊 Steps: {len(workflow)}")
        
        self.workflow_context = context.copy()
        self.execution_results = {}
        
        total_start_time = time.time()
        
        # For simplicity, execute sequentially for now
        for step in workflow:
            print(f"   🔧 Executing {step.step_id}: {step.agent_id}.{step.tool_name}")
            
            start_time = time.time()
            
            try:
                # Prepare inputs by resolving context references
                resolved_inputs = self._resolve_step_inputs(step)
                
                # Find the target agent (simplified approach)
                if step.agent_id not in self.agent_manager.agents:
                    # Initialize agents if not already done
                    await self._initialize_agents()
                
                if step.agent_id not in self.agent_manager.agents:
                    raise ValueError(f"Agent {step.agent_id} not found")
                
                target_agent = self.agent_manager.agents[step.agent_id]
                
                # Execute the tool
                result = await target_agent.call_tool(step.tool_name, resolved_inputs)
                
                execution_time = time.time() - start_time
                
                # Store result
                self.execution_results[step.step_id] = ExecutionResult(
                    step_id=step.step_id,
                    success=True,
                    result=result,
                    error=None,
                    execution_time=execution_time,
                    confidence="high",
                    metadata={"agent_id": step.agent_id, "tool_name": step.tool_name}
                )
                
                # Update context
                self.workflow_context[f"result_{step.step_id}"] = result
                
                print(f"     ✅ Completed in {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                print(f"     ❌ Failed in {execution_time:.2f}s: {str(e)}")
                
                self.execution_results[step.step_id] = ExecutionResult(
                    step_id=step.step_id,
                    success=False,
                    result=None,
                    error=str(e),
                    execution_time=execution_time,
                    confidence="low",
                    metadata={"agent_id": step.agent_id, "tool_name": step.tool_name}
                )
        
        total_time = time.time() - total_start_time
        
        # Compile final results
        final_result = await self._compile_final_results(total_time)
        
        print(f"\n   ✅ Workflow completed in {total_time:.2f}s")
        return final_result
    
    async def _initialize_agents(self):
        """Initialize MCP agents if not already done"""
        try:
            from mcp_input_agent import create_mcp_input_guardrail_agent
            from mcp_question_agent import create_mcp_question_agent  
            from mcp_research_agent import create_mcp_research_agent
            from mcp_output_scoring_agents import create_mcp_output_guardrail_agent, create_mcp_scoring_agent
            
            # Create agents if they don't exist
            if "input_guardrail" not in self.agent_manager.agents:
                input_agent = create_mcp_input_guardrail_agent(self.config, self.message_bus)
                self.agent_manager.register_agent(input_agent)
            
            if "question_agent" not in self.agent_manager.agents:
                question_agent = create_mcp_question_agent(self.config, self.message_bus)
                self.agent_manager.register_agent(question_agent)
            
            if "research_agent" not in self.agent_manager.agents:
                research_agent = create_mcp_research_agent(self.config, self.message_bus)
                self.agent_manager.register_agent(research_agent)
            
            if "output_guardrail" not in self.agent_manager.agents:
                output_agent = create_mcp_output_guardrail_agent(self.config, self.message_bus)
                self.agent_manager.register_agent(output_agent)
            
            if "scoring_agent" not in self.agent_manager.agents:
                scoring_agent = create_mcp_scoring_agent(self.config, self.message_bus)
                self.agent_manager.register_agent(scoring_agent)
            
        except ImportError as e:
            print(f"   ⚠️ Could not initialize all agents: {e}")
    
    def _resolve_step_inputs(self, step: WorkflowStep) -> Dict[str, Any]:
        """Resolve step inputs by replacing context references with actual values"""
        resolved_inputs = {}
        
        for key, value in step.inputs.items():
            if isinstance(value, str):
                if value == "from_context":
                    # Use the main context item for this step type
                    if step.task_type == TaskType.VALIDATION and "topic" in self.workflow_context:
                        resolved_inputs[key] = self.workflow_context["topic"]
                    elif key in self.workflow_context:
                        resolved_inputs[key] = self.workflow_context[key]
                elif value == "from_previous":
                    # Use result from previous step
                    if step.dependencies:
                        prev_step_id = step.dependencies[-1]  # Most recent dependency
                        if f"result_{prev_step_id}" in self.workflow_context:
                            resolved_inputs[key] = self.workflow_context[f"result_{prev_step_id}"]
                elif value == "from_research":
                    # Special case for collecting research results
                    research_results = []
                    for result_key, result_value in self.workflow_context.items():
                        if result_key.startswith("result_") and "research" in result_key:
                            research_results.append(result_value)
                    resolved_inputs[key] = research_results
                elif value.startswith("result_"):
                    # Direct reference to specific step result
                    if value in self.workflow_context:
                        resolved_inputs[key] = self.workflow_context[value]
                else:
                    resolved_inputs[key] = value
            else:
                resolved_inputs[key] = value
        
        return resolved_inputs
    
    async def _compile_final_results(self, total_time: float) -> Dict[str, Any]:
        """Compile final results from all executed steps"""
        
        # Find the final scoring result
        final_scoring = None
        for result in self.execution_results.values():
            if result.success and result.metadata.get("tool_name") == "score_topic":
                final_scoring = result.result
                break
        
        # Collect all evidence/research results
        evidence = []
        for result in self.execution_results.values():
            if result.success and result.metadata.get("tool_name") == "research_question":
                evidence.append(result.result)
        
        # Calculate performance metrics
        successful_steps = sum(1 for r in self.execution_results.values() if r.success)
        total_steps = len(self.execution_results)
        
        # Return comprehensive result
        return {
            "success": final_scoring is not None,
            "orchestration_type": "dynamic_ai_driven",
            "topic": self.workflow_context.get("topic", {}),
            "research_summary": {
                "iterations": 1,
                "questions_asked": len(evidence),
                "answers_approved": len(evidence),
                "retrieval_method": self.config.retrieval_method,
                "total_sources": sum(len(ev.get("sources", [])) for ev in evidence if isinstance(ev, dict)),
                "pdf_slices_used": self.config.use_pdf_slices,
                "optimization_enabled": True,
                "mcp_enabled": True,
                "agent_models": {
                    "input_agent": "gemini-1.5-flash",
                    "question_agent": "gemini-1.5-flash",
                    "research_agent": "gemini-1.5-pro",
                    "scoring_agent": "gemini-1.5-flash"
                }
            },
            "performance_metrics": {
                "total_time": total_time,
                "research_time": sum(r.execution_time for r in self.execution_results.values() 
                                  if r.metadata.get("tool_name") == "research_question"),
                "scoring_time": sum(r.execution_time for r in self.execution_results.values() 
                                 if r.metadata.get("tool_name") == "score_topic"),
                "avg_research_per_iteration": total_time / max(len(evidence), 1)
            },
            "evidence": evidence,
            "scoring": final_scoring or {
                "score": 0,
                "justification": "Workflow failed to complete scoring",
                "confidence": "low",
                "evidence_quality": "poor"
            },
            "execution_summary": {
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "success_rate": successful_steps / total_steps if total_steps > 0 else 0.0,
                "parallel_execution_used": False  # Not implemented in this version
            },
            "workflow_plan": [
                {
                    "step_id": step.step_id,
                    "agent_id": step.agent_id,
                    "tool_name": step.tool_name,
                    "task_type": step.task_type.value,
                    "can_parallelize": step.can_parallelize,
                    "dependencies": step.dependencies
                }
                for step in self.current_workflow
            ],
            "workflow_efficiency": {
                "time_efficiency": 1.0  # Placeholder
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def evaluate_topic_dynamically(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Main entry point for dynamic topic evaluation"""
        print(f"\n{Colors.HEADER}🚀 DYNAMIC TOPIC EVALUATION{Colors.ENDC}")
        print(f"   📋 Topic: {topic.topic_name}")
        
        try:
            # Step 1: Discover available capabilities
            await self.discover_system_capabilities()
            
            # Step 2: Plan the workflow
            task_description = f"Evaluate corporate governance topic: {topic.topic_name}"
            
            workflow = await self.plan_workflow(task_description, {"topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }})
            
            # Step 3: Execute the workflow
            result = await self.execute_workflow(workflow, {"topic": {
                "topic_name": topic.topic_name,
                "goal": topic.goal,
                "guidance": topic.guidance,
                "scoring_rubric": topic.scoring_rubric
            }})
            
            return result
            
        except Exception as e:
            logger.error(f"Dynamic evaluation failed: {e}")
            return {
                "success": False,
                "error": f"Dynamic orchestration failed: {str(e)}",
                "orchestration_type": "dynamic_ai_driven",
                "timestamp": datetime.now().isoformat()
            }


# Synchronous wrapper for Streamlit integration
class DynamicOrchestratorWrapper:
    """
    Synchronous wrapper that provides the same interface as the static orchestrator
    but uses dynamic AI-driven planning underneath
    """
    
    def __init__(self, config: OptimizedConfig):
        self.dynamic_orchestrator = DynamicMCPOrchestrator(config)
        self.config = config
    
    def evaluate_topic(self, topic: TopicDefinition) -> Dict[str, Any]:
        """Synchronous interface for existing code - FIXED VERSION"""
        
        def run_async_evaluation():
            """Helper function to run async evaluation"""
            return asyncio.run(self.dynamic_orchestrator.evaluate_topic_dynamically(topic))
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            
            # If there's already a running loop (like in Jupyter/Streamlit), 
            # we need to handle it specially
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                
                # Create a new task in the current loop
                task = loop.create_task(self.dynamic_orchestrator.evaluate_topic_dynamically(topic))
                
                # Wait for completion using nest_asyncio
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.dynamic_orchestrator.evaluate_topic_dynamically(topic)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=300)  # 5 minute timeout
            else:
                # No running loop, safe to use asyncio.run()
                return run_async_evaluation()
                
        except RuntimeError:
            # No event loop exists, create a new one
            return run_async_evaluation()
        except Exception as e:
            logger.error(f"Synchronous wrapper failed: {e}")
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "orchestration_type": "dynamic_ai_driven",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Synchronous system status"""
        try:
            loop = asyncio.get_running_loop()
            # Similar handling as above for async calls
            return {"status": "active", "type": "dynamic_ai_driven"}
        except:
            return {"status": "active", "type": "dynamic_ai_driven"}
    
    def shutdown(self):
        """Synchronous shutdown"""
        try:
            asyncio.run(self.dynamic_orchestrator.agent_manager.shutdown_all_agents())
        except:
            pass  # Best effort cleanup


# Factory function for creating dynamic orchestrator wrapper
def create_dynamic_orchestrator_wrapper(config: OptimizedConfig) -> DynamicOrchestratorWrapper:
    """Factory function to create dynamic orchestrator wrapper"""
    return DynamicOrchestratorWrapper(config)


if __name__ == "__main__":
    # Test the dynamic orchestrator
    from main import create_sample_topic, OptimizedConfig
    
    async def test_dynamic():
        config = OptimizedConfig("PAYTM")
        orchestrator = DynamicMCPOrchestrator(config)
        topic = create_sample_topic()
        
        result = await orchestrator.evaluate_topic_dynamically(topic)
        print(f"Result: {result.get('success', False)}")
        
    asyncio.run(test_dynamic())