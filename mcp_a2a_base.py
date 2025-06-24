"""
MCP/A2A Base Framework for Enterprise Agentic Corporate Governance System
=========================================================================

This module provides the foundational components for converting the system to use
MCP (Model Context Protocol) and A2A (Agent-to-Agent) protocols while maintaining
the same functionality and interfaces.

Key Components:
- MCPAgent: Base class for all MCP-enabled agents
- A2AMessageBus: Communication layer for agent interactions
- MCPTool: Wrapper for agent capabilities as MCP tools
- A2AMessage: Standardized message format
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import weakref

logger = logging.getLogger(__name__)

# Color codes for console output (same as original)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_mcp_action(agent_name: str, tool_name: str, details: str = ""):
    """Print MCP tool execution with clear formatting"""
    print(f"\n{Colors.OKCYAN}🛠️  MCP TOOL: {agent_name.upper()}.{tool_name}{Colors.ENDC}")
    if details:
        print(f"   Details: {details}")

def print_a2a_message(from_agent: str, to_agent: str, message_type: str, details: str = ""):
    """Print A2A message with clear formatting"""
    print(f"\n{Colors.WARNING}📡 A2A MESSAGE: {from_agent} → {to_agent} ({message_type}){Colors.ENDC}")
    if details:
        print(f"   Details: {details}")

@dataclass
class MCPToolSchema:
    """Schema definition for MCP tools"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

@dataclass
class A2AMessage:
    """Standardized message format for agent-to-agent communication"""
    id: str
    from_agent: str
    to_agent: str
    message_type: str  # "request", "response", "notification", "broadcast"
    tool_name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'A2AMessage':
        return cls(**data)

class A2AMessageBus:
    """
    Agent-to-Agent Message Bus for distributed communication
    
    Handles routing, delivery, and event management between agents.
    Supports both synchronous request-response and asynchronous event patterns.
    """
    
    def __init__(self):
        self.agents: Dict[str, 'MCPAgent'] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.event_subscribers: Dict[str, List[str]] = {}  # event_type -> [agent_ids]
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.message_history: List[A2AMessage] = []
        # Track when requests are sent so we can calculate response latency
        self.request_timestamps: Dict[str, A2AMessage] = {}
        self.running = False
        
    def register_agent(self, agent: 'MCPAgent'):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def send_request(self, from_agent: str, to_agent: str, tool_name: str, 
                          params: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Send synchronous request to another agent"""
        
        if to_agent not in self.agents:
            raise ValueError(f"Agent {to_agent} not found")
        
        message_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        message = A2AMessage(
            id=message_id,
            from_agent=from_agent,
            to_agent=to_agent,
            message_type="request",
            tool_name=tool_name,
            params=params,
            correlation_id=correlation_id
        )

        # Track when the request was sent for latency calculations
        self.request_timestamps[correlation_id] = message
        
        print_a2a_message(from_agent, to_agent, "REQUEST", f"Tool: {tool_name}")
        
        # Create future for response
        future = asyncio.Future()
        self.pending_requests[correlation_id] = future
        
        try:
            # Send message to target agent
            await self._deliver_message(message)
            
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(correlation_id, None)
            self.request_timestamps.pop(correlation_id, None)
            raise TimeoutError(f"Request to {to_agent}.{tool_name} timed out after {timeout}s")
        except Exception as e:
            self.pending_requests.pop(correlation_id, None)
            self.request_timestamps.pop(correlation_id, None)
            raise
    
    async def send_response(self, original_message: A2AMessage, result: Any = None, error: str = None):
        """Send response back to requesting agent"""
        
        response = A2AMessage(
            id=str(uuid.uuid4()),
            from_agent=original_message.to_agent,
            to_agent=original_message.from_agent,
            message_type="response",
            result=result,
            error=error,
            correlation_id=original_message.correlation_id
        )
        
        print_a2a_message(response.from_agent, response.to_agent, "RESPONSE", 
                         f"Success: {error is None}")
        
        # Resolve pending request
        if response.correlation_id in self.pending_requests:
            future = self.pending_requests.pop(response.correlation_id)
            if error:
                future.set_exception(Exception(error))
            else:
                future.set_result(result)
        
        await self._deliver_message(response)
    
    async def broadcast_event(self, from_agent: str, event_type: str, data: Dict[str, Any]):
        """Broadcast event to all subscribers"""
        
        message = A2AMessage(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent="*",  # Broadcast indicator
            message_type="broadcast",
            tool_name=event_type,
            params=data
        )
        
        print_a2a_message(from_agent, "*", "BROADCAST", f"Event: {event_type}")
        
        # Send to all subscribers
        subscribers = self.event_subscribers.get(event_type, [])
        for agent_id in subscribers:
            if agent_id != from_agent and agent_id in self.agents:
                # Create a new message for each subscriber instead of using _replace
                subscriber_message = A2AMessage(
                    id=str(uuid.uuid4()),
                    from_agent=from_agent,
                    to_agent=agent_id,
                    message_type="broadcast",
                    tool_name=event_type,
                    params=data
                )
                await self._deliver_message(subscriber_message)
    
    def subscribe_to_event(self, agent_id: str, event_type: str):
        """Subscribe agent to specific event type"""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        
        if agent_id not in self.event_subscribers[event_type]:
            self.event_subscribers[event_type].append(agent_id)
            logger.info(f"Agent {agent_id} subscribed to event: {event_type}")
    
    async def _deliver_message(self, message: A2AMessage):
        """Internal method to deliver message to target agent"""

        # Store in message history
        self.message_history.append(message)

        # If this is a response, record when it was received
        if message.message_type == "response" and message.correlation_id in self.request_timestamps:
            req_msg = self.request_timestamps.pop(message.correlation_id)
            req_msg.response_timestamp = datetime.now().isoformat()
        
        # Deliver to target agent
        if message.to_agent in self.agents:
            target_agent = self.agents[message.to_agent]
            await target_agent._handle_a2a_message(message)
        else:
            logger.warning(f"Target agent {message.to_agent} not found for message {message.id}")

class MCPAgent(ABC):
    """
    Base class for MCP-enabled agents
    
    Provides standardized MCP tool registration and A2A communication capabilities
    while maintaining the same interfaces as the original agents.
    """
    
    def __init__(self, agent_id: str, config: Any, message_bus: A2AMessageBus):
        self.agent_id = agent_id
        self.config = config
        self.message_bus = message_bus
        self.tools: Dict[str, MCPToolSchema] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.running = False
        
        # Register with message bus
        self.message_bus.register_agent(self)
        
        # Setup tools and start agent
        self._register_tools()
        
        logger.info(f"Initialized MCP Agent: {self.agent_id}")
    
    @abstractmethod
    def _register_tools(self):
        """Register MCP tools specific to this agent"""
        pass
    
    def register_tool(self, schema: MCPToolSchema, handler: Callable):
        """Register an MCP tool with its handler"""
        self.tools[schema.name] = schema
        self.tool_handlers[schema.name] = handler
        logger.info(f"Registered tool {schema.name} for agent {self.agent_id}")
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool locally"""
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Tool {tool_name} not found in agent {self.agent_id}")
        
        print_mcp_action(self.agent_id, tool_name, f"Params: {list(params.keys())}")
        
        start_time = time.time()
        try:
            handler = self.tool_handlers[tool_name]
            
            # Call handler (could be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(params)
            else:
                result = handler(params)
            
            execution_time = time.time() - start_time
            print(f"   ✅ Tool executed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Tool failed in {execution_time:.3f}s: {str(e)}")
            raise
    
    async def call_remote_tool(self, target_agent: str, tool_name: str, 
                              params: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Call a tool on another agent via A2A"""
        return await self.message_bus.send_request(
            self.agent_id, target_agent, tool_name, params, timeout
        )
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to other agents"""
        await self.message_bus.broadcast_event(self.agent_id, event_type, data)
    
    def subscribe_to_event(self, event_type: str):
        """Subscribe to events from other agents"""
        self.message_bus.subscribe_to_event(self.agent_id, event_type)
    
    async def _handle_a2a_message(self, message: A2AMessage):
        """Handle incoming A2A message"""
        try:
            if message.message_type == "request":
                # Execute tool and send response
                result = await self.call_tool(message.tool_name, message.params or {})
                await self.message_bus.send_response(message, result=result)
                
            elif message.message_type == "broadcast":
                # Handle broadcast event
                await self._handle_broadcast_event(message.tool_name, message.params or {})
                
            elif message.message_type == "notification":
                # Handle async notification
                await self._handle_notification(message.tool_name, message.params or {})
                
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")
            if message.message_type == "request":
                await self.message_bus.send_response(message, error=str(e))
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events (override in subclasses)"""
        pass
    
    async def _handle_notification(self, notification_type: str, data: Dict[str, Any]):
        """Handle notifications (override in subclasses)"""
        pass
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for MCP discovery"""
        return [tool.to_dict() for tool in self.tools.values()]
    
    async def shutdown(self):
        """Cleanup and shutdown agent"""
        self.running = False
        self.message_bus.unregister_agent(self.agent_id)
        logger.info(f"Shutdown agent: {self.agent_id}")

class MCPAgentManager:
    """
    Manager for coordinating multiple MCP agents
    
    Provides discovery, lifecycle management, and system-wide coordination.
    """
    
    def __init__(self):
        self.message_bus = A2AMessageBus()
        self.agents: Dict[str, MCPAgent] = {}
        self.running = False
    
    def register_agent(self, agent: MCPAgent):
        """Register an agent with the manager"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent manager registered: {agent.agent_id}")
    
    async def call_agent_tool(self, agent_id: str, tool_name: str, 
                             params: Dict[str, Any]) -> Any:
        """Call a tool on a specific agent"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        return await agent.call_tool(tool_name, params)
    
    async def broadcast_system_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast system-wide event"""
        await self.message_bus.broadcast_event("system", event_type, data)
    
    def get_agent_capabilities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all agent capabilities for discovery"""
        capabilities = {}
        for agent_id, agent in self.agents.items():
            capabilities[agent_id] = agent.get_tool_schemas()
        return capabilities
    
    async def start_all_agents(self):
        """Start all registered agents"""
        self.running = True
        for agent in self.agents.values():
            agent.running = True
        logger.info("Started all agents")
    
    async def shutdown_all_agents(self):
        """Shutdown all agents gracefully"""
        self.running = False
        for agent in self.agents.values():
            await agent.shutdown()
        logger.info("Shutdown all agents")

# Global agent manager instance
agent_manager = MCPAgentManager()

# Utility functions for backward compatibility
async def call_agent_tool(agent_id: str, tool_name: str, params: Dict[str, Any]) -> Any:
    """Global function to call agent tools"""
    return await agent_manager.call_agent_tool(agent_id, tool_name, params)

def get_agent_manager() -> MCPAgentManager:
    """Get the global agent manager"""
    return agent_manager
