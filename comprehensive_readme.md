# 🚀 MCP/A2A Enterprise Agentic Corporate Governance System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](README.md)

**Next-Generation Corporate Governance Analysis System** powered by Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication protocols.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📖 Detailed Usage](#-detailed-usage)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing & Validation](#-testing--validation)
- [📊 Performance Optimization](#-performance-optimization)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

---

## 🎯 Overview

This system provides **enterprise-grade automated analysis** of corporate governance documents using a multi-agent architecture. It evaluates governance topics against custom scoring rubrics by intelligently researching corporate documents and providing detailed, source-cited analysis.

### 🌟 What Makes This Special

- **🛠️ MCP Protocol Integration**: Standardized tool interfaces for maximum interoperability
- **📡 A2A Communication**: Real-time agent coordination and message passing
- **⚡ Lightning Fast**: Pre-computed document processing with intelligent caching
- **🎯 Intelligent Research**: Multi-method retrieval with automatic fallback mechanisms
- **📊 Enterprise Ready**: Comprehensive monitoring, error handling, and scalability

---

## ✨ Key Features

### 🤖 Multi-Agent Architecture
- **Input Guardrail Agent**: Validates topic definitions and rubrics
- **Question Agent**: Generates strategic research questions
- **Research Agent**: Performs high-speed document analysis
- **Output Guardrail Agent**: Validates answer quality
- **Scoring Agent**: Provides final evaluation with justification

### 🔍 Advanced Document Processing
- **Multiple Retrieval Methods**: Hybrid, BM25, Vector, and Direct processing
- **PDF Slice Reconstruction**: Intelligent page extraction for context
- **Semantic Search**: Pre-computed embeddings for instant similarity search
- **Intelligent Fallback**: Automatic method switching when needed

### 📈 Performance Optimizations
- **Pre-computed Chunks**: All document processing done once at startup
- **Cached Embeddings**: Vector computations cached for instant reuse
- **Fast Retrievers**: All search indexes pre-built for millisecond queries
- **Smart Caching**: Automatic cache invalidation and management

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP/A2A ORCHESTRATOR                    │
│                     (main_mcp.py)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │     A2A MESSAGE BUS       │
        │   (Real-time comms)       │
        └─────────────┬─────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
┌───▼───┐     ┌───▼───┐     ┌───▼───┐     ┌───▼───┐     ┌───▼───┐
│Input  │     │Question│    │Research│    │Output │     │Scoring│
│Guard  │     │Agent   │    │Agent   │    │Guard  │     │Agent  │
│MCP    │     │MCP     │    │MCP     │    │MCP    │     │MCP    │
└───────┘     └───────┘     └───────┘     └───────┘     └───────┘
```

### 📁 File Structure

```
production/
├── main_mcp.py                    # 🚀 MAIN ENTRY POINT
├── main.py                        # 🏗️ Legacy components & shared utilities
├── test_agentic.py               # 🧪 Simple testing system
├── mcp_a2a_base.py               # 🔧 MCP/A2A framework foundation
├── mcp_input_agent.py            # 🛡️ Input validation agent
├── mcp_question_agent.py         # ❓ Question generation agent
├── mcp_research_agent.py         # 🔍 Document research agent
├── mcp_output_scoring_agents.py  # 📊 Output validation & scoring
├── mcp_orchestrator.py           # 🎯 Complete orchestration system
├── requirements_mcp.txt          # 📦 Production dependencies
└── data/
    └── {COMPANY}/
        ├── 98_data/              # 📄 PDF documents go here
        ├── 97_cache/             # 💾 Cached chunks & embeddings
        └── 96_results/           # 📈 Evaluation results
```

---

## 📦 Installation

### System Requirements
- **Python**: 3.8+ (recommended: 3.9-3.11)
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ for models and cache
- **OS**: Windows, macOS, Linux

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd AgenticEval_mcp
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements_mcp.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Step 5: Optional - Install Ollama (for local models)
```bash
# Install Ollama from https://ollama.ai/
# Then pull models:
ollama pull llama3
ollama pull llama3.1
```

### Step 6: Verify Installation
```bash
python main_mcp.py --help
```

---

## 🚀 Quick Start

### 1. Prepare Your Documents
Place your PDF documents in the data directory:
```bash
mkdir -p data/YOUR_COMPANY/98_data/
# Copy your PDFs to data/YOUR_COMPANY/98_data/
```

### 2. Run Basic Evaluation
```bash
# Default evaluation with sample topic
python main_mcp.py

# Test specific retrieval method
python main_mcp.py --method hybrid

# Test all methods
python main_mcp.py --test-all
```

### 3. Quick Test with Simple System
```bash
# Run simplified test version
python test_agentic.py
```

---

## 📖 Detailed Usage

### Command Line Interface

#### Basic Commands
```bash
# Default MCP evaluation
python main_mcp.py

# Test specific retrieval method
python main_mcp.py --method [hybrid|bm25|vector|direct]

# Test all retrieval methods
python main_mcp.py --test-all

# Performance comparison and analysis
python main_mcp.py --test-performance

# Demonstrate MCP capabilities
python main_mcp.py --demo-capabilities

# Run in async mode (for advanced users)
python main_mcp.py --async

# Show help
python main_mcp.py --help
```

#### Method Selection
- **`hybrid`** (Default): Combines BM25 + vector search (best overall performance)
- **`bm25`**: Keyword-based search (fast, good for exact terms)
- **`vector`**: Semantic similarity search (best for conceptual queries)
- **`direct`**: Full document processing (most comprehensive, slower)

### Python API Usage

#### Basic Integration
```python
from main_mcp import OptimizedAgenticOrchestrator, OptimizedConfig
from main import TopicDefinition, create_sample_topic

# Setup configuration
config = OptimizedConfig("YOUR_COMPANY")
config.retrieval_method = "hybrid"

# Create orchestrator
orchestrator = OptimizedAgenticOrchestrator(config)

# Create or load topic
topic = create_sample_topic()  # or define your own

# Run evaluation
result = orchestrator.evaluate_topic(topic)

if result["success"]:
    print(f"Score: {result['scoring']['score']}/2")
    print(f"Confidence: {result['scoring']['confidence']}")
else:
    print(f"Error: {result['error']}")

# Cleanup
orchestrator.shutdown()
```

#### Advanced Async Usage
```python
import asyncio
from main_mcp import MCPAgenticOrchestrator

async def advanced_evaluation():
    config = OptimizedConfig("YOUR_COMPANY")
    orchestrator = MCPAgenticOrchestrator(config)
    
    try:
        # Direct agent tool calls
        validation = await orchestrator.input_guardrail_agent.call_tool(
            "validate_topic_definition", {"topic": topic_data}
        )
        
        # Full evaluation
        result = await orchestrator.evaluate_topic(topic)
        return result
        
    finally:
        await orchestrator.shutdown()

# Run async evaluation
result = asyncio.run(advanced_evaluation())
```

### Topic Definition Format

```python
from main import TopicDefinition

topic = TopicDefinition(
    topic_name="Board Independence",
    goal="Assess if the board has directors with permanent board seats",
    guidance="""
    Look for the corporate governance report. Find reappointment dates 
    for each board member. If reappointment date is not provided or 
    older than 5 years, check appointment date. Mark directors as 
    permanent if both dates are older than 5 years.
    """,
    scoring_rubric={
        "0": "Any directors are permanent and not lender representatives",
        "1": "Permanent directors but they are lender representatives", 
        "2": "All directors are non-permanent"
    }
)
```

---

## ⚙️ Configuration

### Basic Configuration

```python
from main import OptimizedConfig

config = OptimizedConfig("YOUR_COMPANY")

# Retrieval method
config.retrieval_method = "hybrid"  # hybrid|bm25|vector|direct

# Document processing
config.chunk_size = 1000           # Characters per chunk
config.chunk_overlap = 200         # Overlap between chunks
config.use_pdf_slices = True       # Enable PDF reconstruction

# Performance settings
config.max_iterations = 3          # Research iterations
config.max_chunks_for_query = None # No limit on chunks
config.force_recompute = False     # Use cached data
```

### Advanced Configuration

#### Model Assignment
```python
# Assign specific models to agents
config.agent_llms = {
    "input_agent": "gemini-1.5-flash",      # Fast validation
    "question_agent": "gemini-1.5-flash",   # Quick questions
    "research_agent": "gemini-1.5-pro",     # High-quality analysis
    "scoring_agent": "gemini-1.5-flash"     # Fast scoring
}

# Set agent-specific temperatures
config.agent_temperatures = {
    "input_agent": 0.1,        # Very consistent
    "question_agent": 0.3,     # Some creativity
    "research_agent": 0.2,     # Factual
    "scoring_agent": 0.1       # Highly consistent
}
```

#### Retrieval Tuning
```python
# Hybrid search weights
config.bm25_weight = 0.4           # Keyword search weight
config.vector_weight = 0.6         # Semantic search weight

# Similarity thresholds
config.similarity_threshold = 0.1   # Vector similarity cutoff
config.bm25_score_threshold = 0.0   # BM25 score cutoff

# PDF processing
config.page_buffer = 1             # Pages around relevant pages
config.max_pdf_size_mb = 20        # Max PDF size for processing
```

#### Intelligent Fallback
```python
# Fallback configuration
config.auto_fallback_to_direct = True     # Enable automatic fallback
config.progressive_escalation = True      # Try docs one-by-one
config.max_direct_documents = 3           # Max docs in fallback

# Fallback trigger keywords
config.fallback_keywords = [
    "no information", "not available", "cannot find",
    "no mention", "not specified", "insufficient information"
]
```

### Directory Structure Configuration

```python
# Default structure
config.base_path = "./data/YOUR_COMPANY/"
config.data_path = "./data/YOUR_COMPANY/98_data/"      # PDFs here
config.cache_path = "./data/YOUR_COMPANY/97_cache/"    # Cache here
# Results saved to: "./data/YOUR_COMPANY/96_results/"

# Custom structure
config = OptimizedConfig("COMPANY", base_path="/custom/path/")
```

---

## 🧪 Testing & Validation

### Built-in Testing

```bash
# Test all retrieval methods
python main_mcp.py --test-all

# Performance benchmarking
python main_mcp.py --test-performance

# MCP capabilities demo
python main_mcp.py --demo-capabilities

# Simple system test
python test_agentic.py
```

### Custom Testing

```python
# Test specific configuration
config = OptimizedConfig("TEST_COMPANY")
config.retrieval_method = "vector"
config.force_recompute = True  # Ignore cache

orchestrator = OptimizedAgenticOrchestrator(config)
result = orchestrator.evaluate_topic(your_topic)

# Validate results
assert result["success"]
assert result["scoring"]["score"] in [0, 1, 2]
assert len(result["evidence"]) > 0
```

### Validation Checklist

- [ ] Documents placed in correct directory
- [ ] API keys configured properly
- [ ] Python virtual environment activated
- [ ] All dependencies installed
- [ ] Test evaluation completes successfully
- [ ] Results saved to 96_results directory

---

## 📊 Performance Optimization

### Speed Optimizations

#### 1. **Pre-computation** (Automatic)
- All chunks computed once at startup
- Embeddings cached permanently
- Retrievers pre-built for instant queries

#### 2. **Smart Caching**
```python
# Enable aggressive caching
config.force_recompute = False  # Use all cached data

# Cache management
config.cache_path = "/fast/ssd/cache/"  # Use SSD for cache
```

#### 3. **Retrieval Method Selection**
```python
# For speed: BM25
config.retrieval_method = "bm25"

# For accuracy: Hybrid (default)
config.retrieval_method = "hybrid"

# For comprehensiveness: Direct
config.retrieval_method = "direct"
```

#### 4. **Model Optimization**
```python
# Use faster models for speed
config.agent_llms = {
    "input_agent": "gemini-1.5-flash",
    "question_agent": "gemini-1.5-flash", 
    "research_agent": "gemini-1.5-flash",  # Faster than pro
    "scoring_agent": "gemini-1.5-flash"
}

# Or use local models
config.agent_llms = {
    "research_agent": "llama3"  # Local processing
}
```

### Memory Optimization

```python
# Reduce chunk size for memory efficiency
config.chunk_size = 500
config.chunk_overlap = 100

# Limit concurrent processing
config.max_chunks_for_query = 50

# Use lighter embedding model
config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

### Performance Monitoring

```python
# Results include detailed metrics
result = orchestrator.evaluate_topic(topic)

print(f"Total Time: {result['performance_metrics']['total_time']:.2f}s")
print(f"Research Time: {result['performance_metrics']['research_time']:.2f}s")
print(f"MCP Overhead: {result['performance_metrics']['mcp_overhead']:.2f}s")
print(f"Tools Used: {result['mcp_metrics']['total_tools']}")
print(f"Messages: {result['mcp_metrics']['message_statistics']['total_messages']}")
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **Import Errors**
```bash
# Error: ModuleNotFoundError
pip install -r requirements_mcp.txt
pip install --upgrade langchain langchain-community
```

#### 2. **API Key Issues**
```bash
# Error: API key not found
echo "GOOGLE_API_KEY=your_key_here" > .env

# Verify API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('GOOGLE_API_KEY')[:10] + '...')"
```

#### 3. **Document Processing Errors**
```python
# Check document accessibility
import os
data_path = "./data/YOUR_COMPANY/98_data/"
pdfs = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
print(f"Found PDFs: {pdfs}")

# Test PDF readability
import fitz
doc = fitz.open(os.path.join(data_path, pdfs[0]))
print(f"Pages: {len(doc)}")
doc.close()
```

#### 4. **Memory Issues**
```python
# Reduce memory usage
config.chunk_size = 500
config.max_chunks_for_query = 20
config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

#### 5. **Slow Performance**
```bash
# Check cache status
ls -la data/YOUR_COMPANY/97_cache/

# Force cache rebuild
python main_mcp.py --method hybrid  # With force_recompute = True
```

#### 6. **Async Issues (Jupyter)**
```python
# In Jupyter notebooks
import nest_asyncio
nest_asyncio.apply()

# Then run async code normally
result = await orchestrator.evaluate_topic(topic)
```

### Debugging Tools

#### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run evaluation with detailed logs
result = orchestrator.evaluate_topic(topic)
```

#### System Status Check
```python
# Check system health
status = orchestrator.get_system_status()
print(f"Agents: {status['agent_count']}")
print(f"Tools: {status['total_tools']}")
print(f"Message Bus: {status['message_bus_status']}")
```

#### Manual Component Testing
```python
# Test document processor
from main import OptimizedDocumentProcessor
processor = OptimizedDocumentProcessor(config)
print(f"Documents loaded: {len(processor.page_chunks)}")

# Test individual agents
input_result = await orchestrator.input_guardrail_agent.call_tool(
    "validate_topic_definition", {"topic": topic_data}
)
print(f"Input validation: {input_result}")
```

### Performance Diagnostics

```bash
# Run performance test
python main_mcp.py --test-performance

# Check results
cat data/YOUR_COMPANY/96_results/optimized_evaluations_summary.csv
```

---

## 🤝 Contributing

### Development Setup

1. **Fork & Clone**
```bash
git fork <repo-url>
git clone <your-fork>
cd AgenticEval_mcp
```

2. **Development Environment**
```bash
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements_mcp.txt
pip install pytest black isort mypy  # Dev tools
```

3. **Code Style**
```bash
# Format code
black *.py
isort *.py

# Type checking
mypy main_mcp.py
```

### Testing Your Changes

```bash
# Run all tests
python main_mcp.py --test-all

# Test specific components
python test_agentic.py

# Performance regression test
python main_mcp.py --test-performance
```

### Submitting Changes

1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and test thoroughly
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Create Pull Request

---

## 📈 System Capabilities

### Supported Document Types
- ✅ PDF documents (corporate reports, governance documents)
- ✅ Multi-page documents with complex layouts
- ✅ Documents up to 20MB (configurable)
- ✅ Multiple documents per evaluation

### Analysis Capabilities
- 🎯 **Custom Topic Evaluation**: Define your own governance topics
- 📊 **Flexible Scoring Rubrics**: 0-2 point scales with custom criteria
- 🔍 **Source Citation**: All answers include page numbers and sources
- 📈 **Evidence Quality Assessment**: Confidence scoring for all findings
- 🚀 **Fast Performance**: Sub-second query response times

### Integration Options
- 🔌 **Python API**: Direct integration into existing systems
- 📡 **MCP Protocol**: Standardized tool interfaces
- 🌐 **A2A Communication**: Real-time agent coordination
- 📊 **JSON Output**: Machine-readable results
- 💾 **CSV Summaries**: Easy data analysis

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 Support

### Getting Help
- 📖 **Documentation**: This README covers 90% of use cases
- 🐛 **Issues**: Report bugs via GitHub Issues
- 💬 **Discussions**: Ask questions in GitHub Discussions
- 📧 **Email**: Contact maintainers for enterprise support

### Quick Support Checklist
Before asking for help, please:
- [ ] Check this README thoroughly
- [ ] Verify your Python version (3.8+)
- [ ] Confirm all dependencies are installed
- [ ] Test with the simple system: `python test_agentic.py`
- [ ] Check that your API keys are configured
- [ ] Verify your documents are in the correct directory

---

**🎉 You're now ready to use the MCP/A2A Enterprise Agentic Corporate Governance System!**

Start with: `python main_mcp.py` and explore the powerful capabilities of modern agent-based document analysis.