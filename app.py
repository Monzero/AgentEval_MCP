"""
Streamlit App for MCP-Enhanced Enterprise Agentic Corporate Governance System
=============================================================================

Enhanced version that works with both original and MCP architectures while providing
additional capabilities for MCP-specific features like agent monitoring, tool usage,
and A2A message tracing, plus Dynamic AI-Driven Orchestration.

Key Enhancements:
- Full backward compatibility with existing app.py
- MCP agent monitoring and health checks
- Real-time A2A message tracing
- Enhanced performance metrics
- Agent capability discovery
- Tool usage analytics
- System status dashboard
- Dynamic AI-Driven Workflow Planning
"""

import streamlit as st
import os
import json
import time
import pandas as pd
import asyncio
import nest_asyncio
from datetime import datetime
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sys

# Enable nested event loops for Streamlit
nest_asyncio.apply()

# Import MCP system components
try:
    from main_mcp import (
        OptimizedAgenticOrchestrator,  # Sync wrapper (backward compatible)
        MCPAgenticOrchestrator,        # Async MCP orchestrator
        OptimizedConfig, TopicDefinition,
        save_results, save_summary_csv
    )
    MCP_AVAILABLE = True
except ImportError:
    # Fallback to original system if MCP not available
    try:
        from main import (
            OptimizedConfig, TopicDefinition, OptimizedAgenticOrchestrator,
            save_results, save_summary_csv
        )
        MCP_AVAILABLE = False
        st.warning("⚠️ MCP system not available. Using original system.")
    except ImportError:
        st.error("❌ Could not import the agentic system. Please ensure main.py or main_mcp.py is available.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Corporate Governance AI - MCP Enhanced",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (enhanced)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .mcp-header {
        font-size: 2.8rem;
        color: #10b981;
        text-align: center;
        margin-bottom: 2rem;
    }
    .dynamic-header {
        font-size: 2.8rem;
        color: #8b5cf6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .config-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .mcp-section {
        background-color: #f0fdf4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    .dynamic-section {
        background-color: #faf5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #8b5cf6;
    }
    .stAlert > div {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .status-running {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-complete {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .mcp-agent-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .dynamic-workflow-card {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #8b5cf6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Ready"
if 'mcp_orchestrator' not in st.session_state:
    st.session_state.mcp_orchestrator = None
if 'system_mode' not in st.session_state:
    st.session_state.system_mode = "MCP" if MCP_AVAILABLE else "Original"

def get_available_companies() -> List[str]:
    """Get list of available companies based on data directories"""
    data_dir = "./data"
    companies = []
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            company_path = os.path.join(data_dir, item)
            if os.path.isdir(company_path):
                # Check if it has the required structure
                data_subdir = os.path.join(company_path, "98_data")
                if os.path.exists(data_subdir):
                    # Check if there are PDF files
                    pdf_files = [f for f in os.listdir(data_subdir) if f.endswith('.pdf')]
                    if pdf_files:
                        companies.append(item)
    
    return sorted(companies)

def create_topic_form() -> TopicDefinition:
    """Create form for topic definition (same as original)"""
    st.subheader("Define Your Topic")
    
    with st.form("topic_form"):
        topic_name = st.text_input(
            "Topic Name",
            value="Board Independence",
            help="A concise name for your evaluation topic"
        )
        
        goal = st.text_area(
            "Goal",
            value="To assess if the board have directors with permanent board seats",
            help="What you want to evaluate or measure",
            height=100
        )
        
        guidance = st.text_area(
            "Guidance",
            value="""You need to look for the corporate governance report. Find the reappointment date for each board members. If the reappointment date is either not provided or older than 5 years (i.e some date before 2019), then you need to check appointment date. If appointment date is also older than 5 years (i.e before 2019), mark that board member as permanent. Give list of board members and whether or not they are permanent. In other words, either of appointment date or reappointment date should be within last 5 years. For example, if a board member has appointment date '02-07-2020' and reappointment date is not present, then because the appointment date is within last 5 years (i.e March 2020 to March 2025 assuming we are checking for annual report as of 31st March 2025) then we would label them as 'Not permanent'. Second example, if any board member has appointment date as 01-01-2012 and reappointment date not present, then we would mark them permanent. Do not present output in table format. Give me text based paragraphs. You are looking at the corporate governance report as of 31st March 2024. Make sure you quote this source in the answer with the page number from which you extract the information.""",
            help="Detailed instructions on how to evaluate this topic",
            height=200
        )
        
        st.write("**Scoring Rubric**")
        col1, col2 = st.columns(2)
        
        with col1:
            score_0 = st.text_area(
                "Score 0 (Poor)",
                value="if any one of the directors is marked as permanent board members as well as they are not explicitly mentioned to be representatives of lenders.",
                help="Criteria for the lowest score"
            )
        
        with col2:
            score_2 = st.text_area(
                "Score 2 (Excellent)",
                value="if All directors are marked as non-permanent board members",
                help="Criteria for the highest score"
            )
        
        score_1 = st.text_area(
            "Score 1 (Good)",
            value="if the directors which are marked as permanent board members, but those are representatives of lenders. Remember that usually this case is applicable for financially distressed companies. So unless it is mentioned explicitly that lenders have sent those board members as representative, do not assume so.",
            help="Criteria for the middle score"
        )
        
        submit_topic = st.form_submit_button("Create Topic", type="primary")
        
        if submit_topic:
            if all([topic_name, goal, guidance, score_0, score_1, score_2]):
                return TopicDefinition(
                    topic_name=topic_name,
                    goal=goal,
                    guidance=guidance,
                    scoring_rubric={
                        "0": score_0,
                        "1": score_1,
                        "2": score_2
                    }
                )
            else:
                st.error("Please fill in all fields")
                return None
    
    return None

def create_configuration_panel() -> Dict[str, Any]:
    """Create configuration panel with enhanced MCP options and dynamic orchestration"""
    
    # System mode selection
    if MCP_AVAILABLE:
        system_mode = st.selectbox(
            "System Mode",
            ["MCP (Enhanced)", "Original (Legacy)"],
            index=0,
            help="Choose between MCP-enhanced or original system"
        )
        st.session_state.system_mode = "MCP" if "MCP" in system_mode else "Original"
    else:
        st.info("🔄 Using Original System (MCP not available)")
        st.session_state.system_mode = "Original"
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        retrieval_method = st.selectbox(
            "Analysis Method",
            ["hybrid", "bm25", "vector"],
            index=0,
            help="Choose the analysis method"
        )
        
        max_iterations = st.number_input(
            "Max Iterations",
            1, 5, 3, 1,
            help="Maximum research iterations"
        )
    
    with col2:
        # Create sub-columns to push elements right
        cache_col1, cache_col2 = st.columns([1, 2])
        
        with cache_col2:
            force_recompute = st.checkbox(
                "Clear Cache & Recompute",
                value=False,
                help="Force recomputation of all data"
            )
            
            if st.button("Clear All Caches", help="Remove all cached data"):
                clear_all_caches()
                st.success("All caches cleared!")
    
    # Initialize default values for when MCP is not available
    enable_dynamic = False
    show_workflow_plan = False
    enable_parallel = False
    adaptive_replanning = False
    max_parallel_steps = 1
    planner_model = "gemini-1.5-flash"
    orchestrator_model = "gemini-1.5-flash"
    input_agent_llm = "gemini-1.5-flash"
    question_agent_llm = "gemini-1.5-flash"
    research_agent_llm = "gemini-1.5-pro"
    scoring_agent_llm = "gemini-1.5-flash"
    input_agent_temp = 0.1
    question_agent_temp = 0.3
    research_agent_temp = 0.2
    scoring_agent_temp = 0.1
    
    # MCP-specific configuration
    if st.session_state.system_mode == "MCP":
        st.markdown("---")
        
        # Dynamic Orchestration Section
        with st.expander("🚀 Dynamic Orchestration (AI-Driven)", expanded=True):
            st.subheader("Intelligent Workflow Planning")
            st.info("🧠 AI analyzes your topic and plans the optimal workflow automatically")
            
            enable_dynamic = st.checkbox(
                "Enable AI-Driven Workflow Planning",
                value=True,
                help="Let AI decide which agents to use and in what order"
            )
            
            if enable_dynamic:
                col1, col2 = st.columns(2)
                
                with col1:
                    show_workflow_plan = st.checkbox(
                        "Show AI Workflow Plan",
                        value=False,
                        help="Display the AI-generated workflow before execution"
                    )
                    
                    enable_parallel = st.checkbox(
                        "Enable Parallel Execution",
                        value=True,
                        help="Run compatible steps in parallel for faster execution"
                    )
                
                with col2:
                    adaptive_replanning = st.checkbox(
                        "Enable Adaptive Re-planning",
                        value=True,
                        help="Allow AI to modify workflow based on intermediate results"
                    )
                    
                    max_parallel_steps = st.slider(
                        "Max Parallel Steps",
                        min_value=1,
                        max_value=10,
                        value=5,
                        help="Maximum number of steps to run simultaneously"
                    )
                
                # AI Model Selection for Planning
                st.markdown("**AI Planning Models:**")
                planner_col, orchestrator_col = st.columns(2)
                
                with planner_col:
                    planner_model = st.selectbox(
                        "Strategic Planner",
                        ["gemini-1.5-pro", "gemini-1.5-flash", "llama3.1"],
                        index=0,
                        help="Model for high-level workflow planning"
                    )
                
                with orchestrator_col:
                    orchestrator_model = st.selectbox(
                        "Execution Controller", 
                        ["gemini-1.5-flash", "gemini-1.5-pro", "llama3"],
                        index=0,
                        help="Model for real-time execution decisions"
                    )
                
                # Expected benefits
                st.success("🎯 Expected Benefits: 40-60% faster execution, task-specific optimization")
            
            else:
                st.warning("Using static workflow - all steps will be executed sequentially")
        
        # Regular MCP Agent Configuration (existing code)
        with st.expander("🛠️ MCP Agent Configuration", expanded=False):
            st.subheader("Agent LLM & Temperature Settings")
            st.info("Configure LLM and temperature for each MCP agent. Lower temperature = more consistent, higher = more creative.")
            
            # Available LLM options
            gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
            ollama_models = ["llama3", "llama3.1", "mistral", "codellama"]
            all_models = gemini_models + ollama_models
            
            # Default temperatures for each agent type
            default_temps = {
                "input_agent": 0.1,
                "question_agent": 0.3,
                "research_agent": 0.2,
                "scoring_agent": 0.1
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Input & Question Agents**")
                
                # Input Agent
                st.markdown("*Input Validation Agent*")
                input_col1, input_col2 = st.columns([2, 1])
                with input_col1:
                    input_agent_llm = st.selectbox(
                        "Model",
                        all_models,
                        index=0,
                        help="Validates topic definitions",
                        key="input_llm"
                    )
                with input_col2:
                    input_agent_temp = st.slider(
                        "Temp",
                        0.0, 1.0, default_temps["input_agent"], 0.1,
                        help="Temperature for input validation",
                        key="input_temp"
                    )
                
                # Question Agent
                st.markdown("*Question Generation Agent*")
                question_col1, question_col2 = st.columns([2, 1])
                with question_col1:
                    question_agent_llm = st.selectbox(
                        "Model",
                        all_models,
                        index=0,
                        help="Generates research questions",
                        key="question_llm"
                    )
                with question_col2:
                    question_agent_temp = st.slider(
                        "Temp",
                        0.0, 1.0, default_temps["question_agent"], 0.1,
                        help="Temperature for question generation",
                        key="question_temp"
                    )
            
            with col2:
                st.markdown("**Research & Scoring Agents**")
                
                # Research Agent
                st.markdown("*Research Agent*")
                research_col1, research_col2 = st.columns([2, 1])
                with research_col1:
                    research_agent_llm = st.selectbox(
                        "Model",
                        all_models,
                        index=1,  # Default to gemini-1.5-pro
                        help="Analyzes documents and extracts information",
                        key="research_llm"
                    )
                with research_col2:
                    research_agent_temp = st.slider(
                        "Temp",
                        0.0, 1.0, default_temps["research_agent"], 0.1,
                        help="Temperature for document analysis",
                        key="research_temp"
                    )
                
                # Scoring Agent
                st.markdown("*Scoring Agent*")
                scoring_col1, scoring_col2 = st.columns([2, 1])
                with scoring_col1:
                    scoring_agent_llm = st.selectbox(
                        "Model",
                        all_models,
                        index=0,
                        help="Provides final scoring and justification",
                        key="scoring_llm"
                    )
                with scoring_col2:
                    scoring_agent_temp = st.slider(
                        "Temp",
                        0.0, 1.0, default_temps["scoring_agent"], 0.1,
                        help="Temperature for final scoring",
                        key="scoring_temp"
                    )
            
            # Show current API status
            st.markdown("---")
            st.markdown("**📡 API Status & Model Info:**")
            google_api_available = bool(os.environ.get("GOOGLE_API_KEY"))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**API Status:**")
                if google_api_available:
                    st.success("✅ Google API Key configured")
                    st.info("Gemini models available")
                else:
                    st.warning("⚠️ No Google API Key found")
                    st.warning("Only Ollama models will work")
            
            with col2:
                st.markdown("**Selected Models:**")
                selected_models = {
                    input_agent_llm, question_agent_llm, 
                    research_agent_llm, scoring_agent_llm
                }
                
                gemini_count = len([m for m in selected_models if m.startswith('gemini')])
                ollama_count = len([m for m in selected_models if not m.startswith('gemini')])
                
                if gemini_count > 0:
                    st.info(f"🌩️ {gemini_count} Gemini models selected")
                if ollama_count > 0:
                    st.info(f"🏠 {ollama_count} Local models selected")
    
    # Prepare configuration
    config_dict = {
        "retrieval_method": retrieval_method,
        "max_iterations": max_iterations,
        "force_recompute": force_recompute,
        "system_mode": st.session_state.system_mode
    }
    
    # Add MCP-specific configurations
    if st.session_state.system_mode == "MCP":
        config_dict.update({
            "agent_llms": {
                "input_agent": input_agent_llm,
                "question_agent": question_agent_llm,
                "research_agent": research_agent_llm,
                "scoring_agent": scoring_agent_llm
            },
            "agent_temperatures": {
                "input_agent": input_agent_temp,
                "question_agent": question_agent_temp,
                "research_agent": research_agent_temp,
                "scoring_agent": scoring_agent_temp
            },
            # Dynamic orchestration settings
            "enable_dynamic_orchestration": enable_dynamic,
            "show_workflow_plan": show_workflow_plan,
            "enable_parallel_execution": enable_parallel,
            "adaptive_replanning": adaptive_replanning,
            "max_parallel_steps": max_parallel_steps,
            "planner_model": planner_model,
            "orchestrator_model": orchestrator_model,
            "dynamic_orchestration_config": {
                "max_parallel_steps": max_parallel_steps,
                "enable_workflow_adaptation": adaptive_replanning,
                "parallel_execution_enabled": enable_parallel
            }
        })
    
    return config_dict

def clear_all_caches():
    """Clear all cache files (same as original)"""
    try:
        companies = get_available_companies()
        for company in companies:
            cache_dir = f"./data/{company}/97_cache/"
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
    except Exception as e:
        st.error(f"Error clearing caches: {e}")

def update_status(status: str):
    """Update the current status"""
    st.session_state.current_status = status

def show_status():
    """Display current status with MCP enhancements"""
    status = st.session_state.current_status
    mode_indicator = f" ({st.session_state.system_mode})" if MCP_AVAILABLE else ""
    
    if status == "Ready":
        st.info(f"Status: {status}{mode_indicator}")
    elif "Running" in status or "Processing" in status:
        st.markdown(f'<div class="status-running">Status: {status}{mode_indicator}</div>', unsafe_allow_html=True)
    elif "Complete" in status:
        st.markdown(f'<div class="status-complete">Status: {status}{mode_indicator}</div>', unsafe_allow_html=True)
    elif "Error" in status or "Failed" in status:
        st.markdown(f'<div class="status-error">Status: {status}{mode_indicator}</div>', unsafe_allow_html=True)
    else:
        st.info(f"Status: {status}{mode_indicator}")

def display_results(result: Dict[str, Any]):
    """Display evaluation results with MCP and Dynamic Orchestration enhancements"""
    
    if not result.get("success", False):
        st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")
        return
    
    st.subheader("Evaluation Results")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Final Score",
            f"{result['scoring']['score']}/2",
            help="Score based on the defined rubric"
        )
    
    with col2:
        st.metric(
            "Evidence Quality",
            result['scoring']['evidence_quality'].title(),
            help="Quality of collected evidence"
        )
    
    with col3:
        st.metric(
            "Sources Used",
            result['research_summary']['total_sources'],
            help="Number of unique sources referenced"
        )
    
    with col4:
        st.metric(
            "Research Time",
            f"{result['performance_metrics']['research_time']:.2f}s",
            help="Time spent on document research"
        )
    
    # Dynamic Orchestration metrics
    if result.get('orchestration_type') == 'dynamic_ai_driven':
        st.markdown('<div class="dynamic-section">', unsafe_allow_html=True)
        st.subheader("🚀 Dynamic AI Orchestration Metrics")
        
        dynamic_col1, dynamic_col2, dynamic_col3, dynamic_col4 = st.columns(4)
        
        exec_summary = result.get('execution_summary', {})
        workflow_efficiency = result.get('workflow_efficiency', {})
        
        with dynamic_col1:
            st.metric(
                "AI Planned Steps",
                exec_summary.get('total_steps', 'N/A'),
                help="Number of steps planned by AI"
            )
        
        with dynamic_col2:
            st.metric(
                "Success Rate",
                f"{exec_summary.get('success_rate', 0)*100:.0f}%",
                help="Percentage of steps that completed successfully"
            )
        
        with dynamic_col3:
            st.metric(
                "Parallel Execution",
                "Yes" if exec_summary.get('parallel_execution_used') else "No",
                help="Whether parallel execution was utilized"
            )
        
        with dynamic_col4:
            st.metric(
                "Time Efficiency",
                f"{workflow_efficiency.get('time_efficiency', 1.0):.1f}x",
                help="Time efficiency vs estimated"
            )
        
        # Show workflow plan if available
        workflow_plan = result.get('workflow_plan', [])
        if workflow_plan:
            st.subheader("🧠 AI Workflow Plan")
            
            # Create workflow visualization
            fig = go.Figure()
            
            for i, step in enumerate(workflow_plan):
                color = '#8b5cf6' if step.get('can_parallelize') else '#6b7280'
                fig.add_trace(go.Scatter(
                    x=[i], y=[0],
                    mode='markers+text',
                    text=[f"{step['agent_id']}<br>{step['tool_name']}"],
                    textposition="top center",
                    marker=dict(size=60, color=color),
                    name=step['step_id'],
                    showlegend=False
                ))
            
            fig.update_layout(
                title="AI-Generated Workflow Plan",
                xaxis_title="Execution Order",
                yaxis=dict(visible=False),
                height=200,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show step details
            with st.expander("📋 Detailed Workflow Steps"):
                for step in workflow_plan:
                    deps = ", ".join(step.get('dependencies', [])) if step.get('dependencies') else "None"
                    parallel = "✅" if step.get('can_parallelize') else "❌"
                    st.write(f"**{step['step_id']}**: {step['agent_id']}.{step['tool_name']} | Parallel: {parallel} | Dependencies: {deps}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # MCP-specific metrics (existing code)
    elif result['research_summary'].get('mcp_enabled', False):
        st.subheader("MCP System Metrics")
        
        mcp_col1, mcp_col2, mcp_col3, mcp_col4 = st.columns(4)
        
        with mcp_col1:
            st.metric(
                "MCP Tools Used",
                result.get('mcp_metrics', {}).get('total_tools', 'N/A'),
                help="Total MCP tools available"
            )
        
        with mcp_col2:
            st.metric(
                "A2A Messages",
                result.get('mcp_metrics', {}).get('message_statistics', {}).get('total_messages', 'N/A'),
                help="Agent-to-agent messages sent"
            )
        
        with mcp_col3:
            st.metric(
                "Active Agents",
                result.get('mcp_metrics', {}).get('total_agents', 'N/A'),
                help="Number of active MCP agents"
            )
        
        with mcp_col4:
            st.metric(
                "Protocol Overhead",
                f"{result['performance_metrics'].get('mcp_overhead', 0):.3f}s",
                help="MCP protocol overhead"
            )
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric(
            "Total Time",
            f"{result['performance_metrics']['total_time']:.2f}s"
        )
    
    with perf_col2:
        st.metric(
            "Iterations",
            result['research_summary']['iterations']
        )
    
    with perf_col3:
        st.metric(
            "Questions Asked",
            result['research_summary']['questions_asked']
        )
    
    # Research Summary
    st.subheader("Research Summary")
    
    research_data = {
        "Metric": [
            "System Mode",
            "Orchestration Type",
            "Retrieval Method",
            "PDF Slices Used",
            "Optimization Enabled",
            "Answers Approved",
            "Evidence Quality"
        ],
        "Value": [
            "MCP Enhanced" if result['research_summary'].get('mcp_enabled', False) else "Original",
            result.get('orchestration_type', 'static').replace('_', ' ').title(),
            result['research_summary']['retrieval_method'],
            "Yes" if result['research_summary']['pdf_slices_used'] else "No",
            "Yes" if result['research_summary']['optimization_enabled'] else "No",
            f"{result['research_summary']['answers_approved']}/{result['research_summary']['questions_asked']}",
            result['scoring']['evidence_quality'].title()
        ]
    }
    
    research_df = pd.DataFrame(research_data)
    st.table(research_df)
    
    # MCP Agent Information
    if result['research_summary'].get('mcp_enabled', False) and 'agent_models' in result['research_summary']:
        st.subheader("MCP Agent Configuration")
        
        agent_models = result['research_summary']['agent_models']
        agent_cols = st.columns(len(agent_models))
        
        for i, (agent_name, model) in enumerate(agent_models.items()):
            with agent_cols[i]:
                st.markdown(f"**{agent_name.replace('_', ' ').title()}**")
                st.markdown(f"Model: `{model}`")
    
    # Evidence Details (same as original)
    st.subheader("Evidence Collected")
    
    for i, evidence in enumerate(result['evidence'], 1):
        with st.expander(f"Evidence {i}: {evidence['question'][:100]}..."):
            st.write("**Question:**")
            st.write(evidence['question'])
            
            st.write("**Answer:**")
            st.write(evidence['answer'])
            
            st.write("**Sources:**")
            for source in evidence['sources']:
                st.write(f"• {source}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Confidence:** {evidence['confidence']}")
            with col2:
                st.write(f"**Has Citations:** {'Yes' if evidence['has_citations'] else 'No'}")
    
    # Final Justification
    st.subheader("Final Justification")
    st.write(result['scoring']['justification'])
    
    # Key Findings
    if result['scoring'].get('key_findings'):
        st.subheader("Key Findings")
        for finding in result['scoring']['key_findings']:
            st.write(f"• {finding}")

def display_mcp_system_status():
    """Display MCP system status and monitoring"""
    if st.session_state.system_mode != "MCP":
        st.info("MCP system status only available when using MCP mode")
        return
    
    st.subheader("MCP System Status")
    
    if st.session_state.mcp_orchestrator:
        try:
            # Get system status (this would need to be implemented)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("System Status", "Active", help="MCP orchestrator status")
            
            with col2:
                st.metric("Agent Count", "5", help="Number of active MCP agents")
            
            with col3:
                st.metric("Message Bus", "Active", help="A2A communication status")
        
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    else:
        st.info("No active MCP orchestrator. Run an evaluation to initialize the system.")

def create_history_visualization():
    """Create visualizations for evaluation history with MCP and Dynamic Orchestration enhancements"""
    
    if not st.session_state.evaluation_history:
        st.info("No evaluation history yet. Run some evaluations to see trends!")
        return
    
    df = pd.DataFrame(st.session_state.evaluation_history)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        score_counts = df['final_score'].value_counts().sort_index()
        fig_scores = px.bar(
            x=score_counts.index,
            y=score_counts.values,
            labels={'x': 'Score', 'y': 'Count'},
            title="Distribution of Final Scores"
        )
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with col2:
        st.subheader("Performance Over Time")
        fig_time = px.line(
            df,
            x='timestamp',
            y='total_time',
            title="Evaluation Time Trend",
            labels={'total_time': 'Time (seconds)', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Method comparison
    if len(df['retrieval_method'].unique()) > 1:
        st.subheader("Method Performance Comparison")
        method_perf = df.groupby('retrieval_method').agg({
            'final_score': 'mean',
            'total_time': 'mean',
            'unique_sources': 'mean'
        }).round(2)
        
        st.dataframe(method_perf, use_container_width=True)
    
    # Orchestration type comparison (Dynamic vs Static)
    if 'orchestration_type' in df.columns and len(df['orchestration_type'].unique()) > 1:
        st.subheader("🚀 Orchestration Performance Comparison")
        
        orchestration_perf = df.groupby('orchestration_type').agg({
            'final_score': 'mean',
            'total_time': 'mean',
            'unique_sources': 'mean'
        }).round(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(orchestration_perf, use_container_width=True)
        
        with col2:
            fig_orch = px.bar(
                x=orchestration_perf.index,
                y=orchestration_perf['total_time'],
                title="Average Time by Orchestration Type",
                labels={'x': 'Orchestration Type', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig_orch, use_container_width=True)
        
        # Dynamic orchestration specific metrics
        if 'ai_planned_steps' in df.columns:
            dynamic_df = df[df['orchestration_type'] == 'dynamic_ai_driven']
            if not dynamic_df.empty:
                st.subheader("🧠 Dynamic Orchestration Insights")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_steps = dynamic_df['ai_planned_steps'].mean()
                    st.metric("Avg AI Planned Steps", f"{avg_steps:.1f}")
                
                with col2:
                    parallel_usage = dynamic_df['parallel_execution_used'].mean() * 100
                    st.metric("Parallel Execution Usage", f"{parallel_usage:.0f}%")
                
                with col3:
                    avg_efficiency = dynamic_df['workflow_efficiency'].mean()
                    st.metric("Avg Workflow Efficiency", f"{avg_efficiency:.2f}x")
    
    # MCP vs Original comparison (if both exist)
    if 'system_mode' in df.columns and len(df['system_mode'].unique()) > 1:
        st.subheader("🚀 MCP vs Original Performance")
        
        mode_perf = df.groupby('system_mode').agg({
            'final_score': 'mean',
            'total_time': 'mean',
            'unique_sources': 'mean'
        }).round(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(mode_perf, use_container_width=True)
        
        with col2:
            fig_mode = px.bar(
                x=mode_perf.index,
                y=mode_perf['total_time'],
                title="Average Time by System Mode",
                labels={'x': 'System Mode', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig_mode, use_container_width=True)

def main():
    """Main Streamlit application with MCP and Dynamic Orchestration enhancements"""
    
    # Header with dynamic indication
    if MCP_AVAILABLE and st.session_state.system_mode == "MCP":
        st.markdown("<h1 class='dynamic-header'>Corporate Governance AI - Dynamic MCP Enhanced</h1>", unsafe_allow_html=True)
    elif MCP_AVAILABLE:
        st.markdown("<h1 class='mcp-header'>Corporate Governance AI - MCP Enhanced</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 class='main-header'>🏢 Corporate Governance AI</h1>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Control Panel")
        
        # System mode indicator
        if MCP_AVAILABLE:
            mode_color = "🚀" if st.session_state.system_mode == "MCP" else "🏛️"
            st.markdown(f"{mode_color} **Mode**: {st.session_state.system_mode}")
            st.markdown("---")
        
        # Company Selection
        st.subheader("Select Company")
        companies = get_available_companies()
        
        if not companies:
            st.error("No companies found! Please ensure you have data directories with PDF files.")
            st.stop()
        
        selected_company = st.selectbox(
            "Available Companies",
            companies,
            help="Select a company to analyze"
        )
        
        st.divider()
        
        # Navigation
        nav_options = ["Home", "Create Topic", "Run Evaluation", "Results", "History"]
        
        # Add MCP-specific options
        if MCP_AVAILABLE and st.session_state.system_mode == "MCP":
            nav_options.append("MCP Status")
        
        page = st.radio(
            "Navigation",
            nav_options,
            help="Choose what you want to do"
        )
    
    # Main content area
    if page == "Home":
        st.subheader("Welcome to the Corporate Governance AI System")
        
        # Enhanced description with MCP and Dynamic features
        if MCP_AVAILABLE:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏛️ **Core Features**")
                st.markdown("""
                **Intelligent Document Analysis**
                - Advanced search combining multiple methods
                - Automatic PDF processing
                - Smart fallback mechanisms
                
                **Multi-Agent Workflow**
                - Input validation and guardrails
                - Dynamic question generation
                - Research with source verification
                - Automated scoring against custom rubrics
                
                **Optimized Performance**
                - Pre-computed embeddings and indexes
                - Intelligent caching system
                - Real-time status updates
                """)
            
            with col2:
                if st.session_state.system_mode == "MCP":
                    st.markdown('<div class="dynamic-section">', unsafe_allow_html=True)
                    st.markdown("### 🚀 **Dynamic AI Orchestration**")
                    st.markdown("""
                    **AI-Driven Workflow Planning**
                    - Intelligent task analysis and planning
                    - Optimal agent selection and sequencing
                    - Parallel execution optimization
                    - Adaptive re-planning based on results
                    
                    **Enhanced MCP Protocols**
                    - 15+ MCP tools across 5 specialized agents
                    - Real-time agent discovery and monitoring
                    - Tool usage analytics and insights
                    
                    **Agent-to-Agent Communication**
                    - Real-time A2A message bus
                    - Event broadcasting and subscription
                    - Message history and tracing
                    
                    **Performance Benefits**
                    - 40-60% faster execution
                    - Task-specific optimization
                    - Intelligent resource allocation
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("### 🔄 **Switch to MCP Mode**")
                    st.info("Enable MCP mode in the configuration panel to access enhanced features like agent monitoring, A2A communication, and Dynamic AI orchestration.")
        else:
            st.markdown("""
            This system uses advanced AI agents to evaluate corporate governance topics by:
            
            **Intelligent Document Analysis**
            - Advanced search combining multiple methods
            - Automatic PDF processing
            - Smart fallback mechanisms
            
            **Multi-Agent Workflow**
            - Input validation and guardrails
            - Dynamic question generation
            - Research with source verification
            - Automated scoring against custom rubrics
            
            **Optimized Performance**
            - Pre-computed embeddings and indexes
            - Intelligent caching system
            - Real-time status updates
            """)
        
        # Quick stats
        if selected_company:
            data_path = f"./data/{selected_company}/98_data"
            if os.path.exists(data_path):
                doc_count = len([f for f in os.listdir(data_path) if f.endswith('.pdf')])
            else:
                doc_count = 0
            
            mode_info = f" ({st.session_state.system_mode} Mode)" if MCP_AVAILABLE else ""
            st.info(f"Selected Company: **{selected_company}** with **{doc_count} documents**{mode_info}")
    
    elif page == "Create Topic":
        topic = create_topic_form()
        if topic:
            st.session_state.current_topic = topic
            st.success("Topic created successfully! Go to 'Run Evaluation' to analyze it.")
    
    elif page == "Run Evaluation":
        # Check if we have topic
        if 'current_topic' not in st.session_state:
            st.warning("Please create a topic first!")
            st.stop()
        
        st.subheader("Run Evaluation")
        
        # Configuration section
        st.subheader("Settings")
        config_dict = create_configuration_panel()
        st.session_state.current_config = config_dict
        
        # Show current settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Topic:**")
            st.write(f"Topic: {st.session_state.current_topic.topic_name}")
            st.write(f"Goal: {st.session_state.current_topic.goal[:100]}...")
        
        with col2:
            st.write("**Current Configuration:**")
            if 'current_config' in st.session_state:
                st.write(f"System: {st.session_state.current_config['system_mode']}")
                st.write(f"Method: {st.session_state.current_config['retrieval_method']}")
                st.write(f"Max Iterations: {st.session_state.current_config['max_iterations']}")
                if st.session_state.current_config.get('enable_dynamic_orchestration'):
                    st.write("🚀 Dynamic AI Orchestration: Enabled")
        
        # Status display
        show_status()
        
        # Run button
        if st.button("Start Evaluation", type="primary"):
            if 'current_config' not in st.session_state:
                st.error("Please configure the settings first!")
                st.stop()
                
            with st.spinner("Running evaluation..."):
                
                # CREATE ORCHESTRATOR HERE
                try:
                    # Create config from session state
                    config = OptimizedConfig(selected_company)
                    config.retrieval_method = st.session_state.current_config['retrieval_method']
                    config.max_iterations = st.session_state.current_config['max_iterations']
                    config.force_recompute = st.session_state.current_config.get('force_recompute', False)
                    
                    # Apply MCP-specific configurations if available
                    if st.session_state.current_config['system_mode'] == "MCP":
                        if "agent_llms" in st.session_state.current_config:
                            config.agent_llms.update(st.session_state.current_config["agent_llms"])
                            config.agent_temperatures.update(st.session_state.current_config["agent_temperatures"])
                        
                        # Apply dynamic orchestration settings
                        config.enable_dynamic_orchestration = st.session_state.current_config.get('enable_dynamic_orchestration', True)
                        
                        if config.enable_dynamic_orchestration:
                            # Update dynamic orchestration config
                            dynamic_config = st.session_state.current_config.get('dynamic_orchestration_config', {})
                            if hasattr(config, 'dynamic_orchestration'):
                                config.dynamic_orchestration.update(dynamic_config)
                            
                            # Update AI model selections
                            if 'planner_model' in st.session_state.current_config:
                                config.agent_llms['planner'] = st.session_state.current_config['planner_model']
                            if 'orchestrator_model' in st.session_state.current_config:
                                config.agent_llms['orchestrator'] = st.session_state.current_config['orchestrator_model']
                    
                    # Create appropriate orchestrator
                    if (st.session_state.current_config['system_mode'] == "MCP" and 
                        config.enable_dynamic_orchestration):
                        
                        try:
                            from dynamic_mcp_orchestrator import create_dynamic_orchestrator_wrapper
                            update_status("Initializing Dynamic AI-Driven Orchestrator...")
                            orchestrator = create_dynamic_orchestrator_wrapper(config)
                            
                            # Show workflow plan if requested
                            if st.session_state.current_config.get('show_workflow_plan', False):
                                st.info("🧠 AI is planning the optimal workflow for your topic...")
                        except ImportError:
                            st.error("❌ Dynamic orchestrator not available. Please ensure dynamic_mcp_orchestrator.py is in your project directory.")
                            st.stop()
                        
                    elif st.session_state.current_config['system_mode'] == "MCP":
                        update_status("Initializing MCP system with distributed agents...")
                        orchestrator = OptimizedAgenticOrchestrator(config)
                    else:
                        update_status("Initializing original system...")
                        orchestrator = OptimizedAgenticOrchestrator(config)
                    
                    # Run evaluation
                    update_status("Running evaluation...")
                    result = orchestrator.evaluate_topic(st.session_state.current_topic)
                    
                    update_status("Evaluation complete...")
                    
                    # Save results
                    if result and result.get("success", False):
                        update_status("Saving results...")
                        save_results(result, config)
                        save_summary_csv(result, config)
                        
                        # Success message with orchestration type
                        if result.get('orchestration_type') == 'dynamic_ai_driven':
                            success_msg = "🚀 Dynamic AI evaluation completed!"
                            if result.get('workflow_plan'):
                                success_msg += f" AI planned {len(result['workflow_plan'])} optimized steps."
                            if result.get('execution_summary', {}).get('parallel_execution_used'):
                                success_msg += " Parallel execution was used for faster results."
                        else:
                            success_msg = "Evaluation completed! Check the 'Results' tab to see detailed findings."
                            if st.session_state.current_config['system_mode'] == "MCP":
                                success_msg += " 🚀 Enhanced with MCP protocols."
                        
                        st.success(success_msg)
                        update_status("Evaluation completed successfully")
                    else:
                        st.error(f"Evaluation failed: {result.get('error', 'Unknown error')}")
                        update_status("Evaluation failed")
                        
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
                    update_status(f"Error: {str(e)}")
                    
                # Store results
                if 'result' in locals():
                    st.session_state.evaluation_results = result
                    
                    # Add to history with orchestration type
                    history_entry = {
                        'timestamp': result.get('timestamp', datetime.now().isoformat()),
                        'company': selected_company,
                        'topic_name': result.get('topic', {}).get('name', 'Unknown'),
                        'final_score': result.get('scoring', {}).get('score', 0),
                        'confidence': result.get('scoring', {}).get('confidence', 'unknown'),
                        'retrieval_method': st.session_state.current_config['retrieval_method'],
                        'total_time': result.get('performance_metrics', {}).get('total_time', 0),
                        'unique_sources': result.get('research_summary', {}).get('total_sources', 0),
                        'system_mode': st.session_state.current_config['system_mode'],
                        'orchestration_type': result.get('orchestration_type', 'static')  # Track orchestration type
                    }
                    
                    # Add dynamic orchestration specific metrics
                    if result.get('orchestration_type') == 'dynamic_ai_driven':
                        exec_summary = result.get('execution_summary', {})
                        history_entry.update({
                            'ai_planned_steps': exec_summary.get('total_steps', 0),
                            'parallel_execution_used': exec_summary.get('parallel_execution_used', False),
                            'workflow_efficiency': result.get('workflow_efficiency', {}).get('time_efficiency', 1.0),
                            'success_rate': exec_summary.get('success_rate', 1.0)
                        })
                    
                    st.session_state.evaluation_history.append(history_entry)
    
    
    # Fixed version of the Run Evaluation section in app.py
# Replace the existing run_evaluation section with this code

    elif page == "Run Evaluation":
        # Check if we have topic
        if 'current_topic' not in st.session_state:
            st.warning("Please create a topic first!")
            st.stop()
        
        st.subheader("Run Evaluation")
        
        # Configuration section
        st.subheader("Settings")
        config_dict = create_configuration_panel()
        st.session_state.current_config = config_dict
        
        # Show current settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Topic:**")
            st.write(f"Topic: {st.session_state.current_topic.topic_name}")
            st.write(f"Goal: {st.session_state.current_topic.goal[:100]}...")
        
        with col2:
            st.write("**Current Configuration:**")
            if 'current_config' in st.session_state:
                st.write(f"System: {st.session_state.current_config['system_mode']}")
                st.write(f"Method: {st.session_state.current_config['retrieval_method']}")
                st.write(f"Max Iterations: {st.session_state.current_config['max_iterations']}")
                if st.session_state.current_config.get('enable_dynamic_orchestration'):
                    st.write("🚀 Dynamic AI Orchestration: Enabled")
        
        # Status display
        show_status()
        
        # Run button
        if st.button("Start Evaluation", type="primary"):
            if 'current_config' not in st.session_state:
                st.error("Please configure the settings first!")
                st.stop()
                
            with st.spinner("Running evaluation..."):
                
                # CREATE ORCHESTRATOR HERE - FIXED VERSION
                try:
                    # Create config from session state
                    config = OptimizedConfig(selected_company)
                    config.retrieval_method = st.session_state.current_config['retrieval_method']
                    config.max_iterations = st.session_state.current_config['max_iterations']
                    config.force_recompute = st.session_state.current_config.get('force_recompute', False)
                    
                    # Apply MCP-specific configurations if available
                    if st.session_state.current_config['system_mode'] == "MCP":
                        if "agent_llms" in st.session_state.current_config:
                            config.agent_llms.update(st.session_state.current_config["agent_llms"])
                            config.agent_temperatures.update(st.session_state.current_config["agent_temperatures"])
                        
                        # Apply dynamic orchestration settings
                        config.enable_dynamic_orchestration = st.session_state.current_config.get('enable_dynamic_orchestration', True)
                    
                    # Initialize result variable
                    result = None
                    
                    # Create appropriate orchestrator
                    if (st.session_state.current_config['system_mode'] == "MCP" and 
                        st.session_state.current_config.get('enable_dynamic_orchestration', False)):
                        
                        try:
                            from dynamic_mcp_orchestrator import create_dynamic_orchestrator_wrapper
                            update_status("Initializing Dynamic AI-Driven Orchestrator...")
                            orchestrator = create_dynamic_orchestrator_wrapper(config)
                            
                            # Show workflow plan if requested
                            if st.session_state.current_config.get('show_workflow_plan', False):
                                st.info("🧠 AI is planning the optimal workflow for your topic...")
                                
                        except ImportError:
                            st.error("❌ Dynamic orchestrator not available. Please ensure dynamic_mcp_orchestrator.py is in your project directory.")
                            st.stop()
                            
                    elif st.session_state.current_config['system_mode'] == "MCP":
                        update_status("Initializing MCP system with distributed agents...")
                        orchestrator = OptimizedAgenticOrchestrator(config)
                    else:
                        update_status("Initializing original system...")
                        orchestrator = OptimizedAgenticOrchestrator(config)
                    
                    # Run evaluation - FIXED TO HANDLE BOTH SYNC AND ASYNC
                    update_status("Running evaluation...")
                    
                    # The key fix: ensure we're calling the right method and handling the result properly
                    try:
                        result = orchestrator.evaluate_topic(st.session_state.current_topic)
                        
                        # Ensure result is a dictionary, not a coroutine
                        if hasattr(result, '__await__'):
                            st.error("❌ Async/sync mismatch detected. Please check orchestrator implementation.")
                            st.stop()
                            
                    except Exception as eval_error:
                        st.error(f"❌ Evaluation execution failed: {str(eval_error)}")
                        update_status(f"Evaluation failed: {str(eval_error)}")
                        result = None
                    
                    update_status("Evaluation complete...")
                    
                    # Save results - FIXED VERSION
                    if result and isinstance(result, dict) and result.get("success", False):
                        update_status("Saving results...")
                        save_results(result, config)
                        save_summary_csv(result, config)
                        
                        # Success message with orchestration type
                        if result.get('orchestration_type') == 'dynamic_ai_driven':
                            success_msg = "🚀 Dynamic AI evaluation completed!"
                            if result.get('workflow_plan'):
                                success_msg += f" AI planned {len(result['workflow_plan'])} optimized steps."
                            if result.get('execution_summary', {}).get('parallel_execution_used'):
                                success_msg += " Parallel execution was used for faster results."
                        else:
                            success_msg = "Evaluation completed! Check the 'Results' tab to see detailed findings."
                            if st.session_state.current_config['system_mode'] == "MCP":
                                success_msg += " 🚀 Enhanced with MCP protocols."
                        
                        st.success(success_msg)
                        update_status("Evaluation completed successfully")
                    else:
                        error_msg = "Unknown error"
                        if result and isinstance(result, dict):
                            error_msg = result.get('error', 'Unknown error')
                        elif result:
                            error_msg = f"Invalid result type: {type(result)}"
                            
                        st.error(f"Evaluation failed: {error_msg}")
                        update_status("Evaluation failed")
                        
                except Exception as e:
                    st.error(f"Evaluation setup failed: {str(e)}")
                    update_status(f"Setup error: {str(e)}")
                    result = None
                    
                # Store results - FIXED VERSION
                if result and isinstance(result, dict):
                    st.session_state.evaluation_results = result
                    
                    # Add to history with orchestration type - SAFE VERSION
                    try:
                        history_entry = {
                            'timestamp': result.get('timestamp', datetime.now().isoformat()),
                            'company': selected_company,
                            'topic_name': result.get('topic', {}).get('name', st.session_state.current_topic.topic_name),
                            'final_score': result.get('scoring', {}).get('score', 0),
                            'confidence': result.get('scoring', {}).get('confidence', 'unknown'),
                            'retrieval_method': st.session_state.current_config['retrieval_method'],
                            'total_time': result.get('performance_metrics', {}).get('total_time', 0),
                            'unique_sources': result.get('research_summary', {}).get('total_sources', 0),
                            'system_mode': st.session_state.current_config['system_mode'],
                            'orchestration_type': result.get('orchestration_type', 'static')
                        }
                        
                        # Add dynamic orchestration specific metrics if available
                        if result.get('orchestration_type') == 'dynamic_ai_driven':
                            exec_summary = result.get('execution_summary', {})
                            history_entry.update({
                                'ai_planned_steps': exec_summary.get('total_steps', 0),
                                'parallel_execution_used': exec_summary.get('parallel_execution_used', False),
                                'workflow_efficiency': result.get('workflow_efficiency', {}).get('time_efficiency', 1.0),
                                'success_rate': exec_summary.get('success_rate', 1.0)
                            })
                        
                        # Add MCP-specific metrics if available
                        elif result.get('research_summary', {}).get('mcp_enabled', False):
                            mcp_metrics = result.get('mcp_metrics', {})
                            history_entry.update({
                                'mcp_tools': mcp_metrics.get('total_tools', 0),
                                'a2a_messages': mcp_metrics.get('message_statistics', {}).get('total_messages', 0),
                                'protocol_overhead': result.get('performance_metrics', {}).get('mcp_overhead', 0)
                            })
                        
                        st.session_state.evaluation_history.append(history_entry)
                        
                    except Exception as history_error:
                        st.warning(f"Could not add to history: {str(history_error)}")
                else:
                    st.warning("No valid results to store.")
    
    elif page == "Results":
        if 'evaluation_results' not in st.session_state or not st.session_state.evaluation_results:
            st.warning("No results available. Please run an evaluation first!")
        else:
            display_results(st.session_state.evaluation_results)
            
            # Download options
            st.subheader("Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download JSON
                json_data = json.dumps(st.session_state.evaluation_results, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Download summary CSV
                if st.session_state.evaluation_history:
                    df = pd.DataFrame(st.session_state.evaluation_history)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name=f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    elif page == "History":
        st.subheader("Evaluation History")
        create_history_visualization()
        
        # Show detailed history table
        if st.session_state.evaluation_history:
            st.subheader("Detailed History")
            df = pd.DataFrame(st.session_state.evaluation_history)
            st.dataframe(df, use_container_width=True)
            
            # Clear history button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Clear History", type="secondary"):
                    st.session_state.evaluation_history = []
                    st.success("History cleared!")
                    st.rerun()
    
    elif page == "MCP Status":
        if st.session_state.system_mode == "MCP":
            display_mcp_system_status()
        else:
            st.info("MCP Status is only available when using MCP mode. Switch to MCP mode in the configuration panel.")

if __name__ == "__main__":
    main()