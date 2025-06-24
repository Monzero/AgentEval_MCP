"""
MCP Research Agent - High-Performance Document Research Engine
==============================================================

Converted to use MCP tools and A2A communication while maintaining the same functionality
as the original OptimizedResearchAgent. This agent provides fast document analysis using
pre-computed chunks, embeddings, and intelligent fallback mechanisms.

MCP Tools Provided:
- research_question: Main research tool for answering questions
- search_documents: Fast document search using multiple methods
- extract_information: Extract specific information from documents
- analyze_document_quality: Assess document quality and relevance
- get_document_metadata: Retrieve document metadata and statistics

MCP Resources Provided:
- cached_chunks: Pre-computed document chunks
- embeddings: Pre-computed vector embeddings
- document_indexes: Pre-built search indexes

A2A Events:
- research_started: Broadcast when research begins
- documents_found: Broadcast when relevant documents identified
- research_completed: Broadcast when research finishes
- fallback_triggered: Broadcast when fallback method activated
"""

import asyncio
import json
import logging
import time
import os
import base64
import tempfile
import fitz
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import base framework
from mcp_a2a_base import MCPAgent, MCPToolSchema, A2AMessage, print_mcp_action

# Import original dependencies  
from main import (
    OptimizedConfig, OptimizedDocumentProcessor, Answer, LLMManager, Colors,
    print_llm_interaction, print_retrieval_info
)

logger = logging.getLogger(__name__)

class MCPResearchAgent(MCPAgent):
    """
    MCP-enabled Research Agent
    
    Provides the same high-performance document research functionality as the original
    OptimizedResearchAgent but exposed as MCP tools with A2A communication capabilities.
    """
    
    def __init__(self, config: OptimizedConfig, message_bus):
        # Initialize document processor (same as original)
        self.document_processor = OptimizedDocumentProcessor(config)
        
        # Initialize LLM manager (same as original)
        self.llm_manager = LLMManager(config)
        self.llm = None
        self.current_model = None
        self._setup_llm()
        
        # Initialize as MCP agent
        super().__init__("research_agent", config, message_bus)
        
        # Subscribe to relevant events
        self.subscribe_to_event("question_generated")
        self.subscribe_to_event("evidence_gap_identified")
        self.subscribe_to_event("evaluation_started")
    
    def _setup_llm(self):
        """Setup LLM for document analysis (same as original)"""
        self.llm, self.current_model = self.llm_manager.get_llm("research_agent")
    
    def _register_tools(self):
        """Register MCP tools for document research"""
        
        # Main research tool
        self.register_tool(
            MCPToolSchema(
                name="research_question",
                description="Research a question using optimized document analysis with intelligent fallback",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "max_chunks": {"type": "integer", "minimum": 1, "maximum": 100},
                        "retrieval_method": {
                            "type": "string",
                            "enum": ["hybrid", "bm25", "vector", "direct"],
                            "description": "Override default retrieval method"
                        },
                        "enable_fallback": {"type": "boolean", "default": True},
                        "timeout": {"type": "number", "default": 60.0}
                    },
                    "required": ["question"]
                }
            ),
            self._research_question_tool
        )
        
        # Document search tool
        self.register_tool(
            MCPToolSchema(
                name="search_documents",
                description="Fast semantic and keyword search across all documents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["hybrid", "bm25", "vector"],
                            "default": "hybrid"
                        },
                        "max_results": {"type": "integer", "default": 20},
                        "similarity_threshold": {"type": "number", "default": 0.1}
                    },
                    "required": ["query"]
                }
            ),
            self._search_documents_tool
        )
        
        # Information extraction tool
        self.register_tool(
            MCPToolSchema(
                name="extract_information",
                description="Extract specific information from document chunks or full documents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "extraction_query": {"type": "string"},
                        "document_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific documents to search in"
                        },
                        "extraction_method": {
                            "type": "string",
                            "enum": ["chunks", "pdf_slices", "full_document"],
                            "default": "chunks"
                        }
                    },
                    "required": ["extraction_query"]
                }
            ),
            self._extract_information_tool
        )
        
        # Document quality analysis tool
        self.register_tool(
            MCPToolSchema(
                name="analyze_document_quality",
                description="Assess quality and relevance of documents for a given topic",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic_keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "quality_criteria": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["completeness", "clarity", "relevance", "citations"]
                        }
                    },
                    "required": ["topic_keywords"]
                }
            ),
            self._analyze_document_quality_tool
        )
        
        # Document metadata tool
        self.register_tool(
            MCPToolSchema(
                name="get_document_metadata",
                description="Retrieve metadata and statistics about available documents",
                input_schema={
                    "type": "object",
                    "properties": {
                        "include_statistics": {"type": "boolean", "default": True},
                        "include_chunk_info": {"type": "boolean", "default": False},
                        "document_filter": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter to specific documents"
                        }
                    }
                }
            ),
            self._get_document_metadata_tool
        )
    
    async def _research_question_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Research question using optimized retrieval
        
        Same logic as original research_question but as an MCP tool
        """
        question = params["question"]
        max_chunks = params.get("max_chunks")
        retrieval_method = params.get("retrieval_method", self.config.retrieval_method)
        enable_fallback = params.get("enable_fallback", True)
        timeout = params.get("timeout", 60.0)
        
        if not self.llm:
            print(f"   ❌ No LLM available for research")
            await self.broadcast_event("research_completed", {
                "question": question,
                "success": False,
                "error": "No LLM available"
            })
            return {
                "question": question,
                "answer": "No LLM available for research",
                "sources": [],
                "confidence": "low",
                "has_citations": False
            }
        
        # Broadcast research started
        await self.broadcast_event("research_started", {
            "question": question,
            "method": retrieval_method,
            "max_chunks": max_chunks
        })
        
        # Override config temporarily if needed
        original_method = self.config.retrieval_method
        if retrieval_method != original_method:
            self.config.retrieval_method = retrieval_method
        
        try:
            start_time = time.time()
            
            # Phase 1: Try retrieval-based approach first
            print(f"   🔍 Phase 1: Trying retrieval-based approach...")
            relevant_chunks = await self._find_relevant_chunks_optimized(question["text"], max_chunks)
            retrieval_time = time.time() - start_time
            
            print(f"   ⚡ Retrieval completed in {retrieval_time:.3f}s")
            
            if not relevant_chunks:
                if enable_fallback:
                    print(f"   ❌ No relevant chunks found - will try direct method")
                    return await self._fallback_to_direct_method(question, "No relevant chunks found")
                else:
                    await self.broadcast_event("research_completed", {
                        "question": question,
                        "success": False,
                        "error": "No relevant chunks found"
                    })
                    return {
                        "question": question,
                        "answer": "No relevant information found in documents",
                        "sources": [],
                        "confidence": "low",
                        "has_citations": False
                    }
            
            # Broadcast documents found
            sources = list(set([chunk.metadata.get('source', 'Unknown') for chunk in relevant_chunks]))
            await self.broadcast_event("documents_found", {
                "question": question,
                "chunk_count": len(relevant_chunks),
                "sources": sources,
                "method": retrieval_method
            })
            
            # Try retrieval-based approach
            print(f"   📝 Processing {len(relevant_chunks)} chunks with retrieval method...")
            
            if self.config.use_pdf_slices:
                print(f"   🔀 Using PDF slice reconstruction...")
                answer_text = await self._query_with_pdf_slices_and_check(question, relevant_chunks)
            else:
                print(f"   📝 Using text chunks...")
                answer_text = await self._query_with_chunks_and_check(question, relevant_chunks)
            
            # Check if LLM found sufficient information
            if enable_fallback and self._should_fallback_to_direct(answer_text, question):
                print(f"   🔄 LLM indicates insufficient information - falling back to direct method")
                await self.broadcast_event("fallback_triggered", {
                    "question": question,
                    "reason": "Insufficient information in chunks",
                    "original_method": retrieval_method
                })
                return await self._fallback_to_direct_method(question, "Insufficient information in chunks", relevant_chunks)
            
            # Retrieval was successful
            sources = self._extract_sources_from_chunks(relevant_chunks)
            has_citations = self._has_source_citations(answer_text)
            confidence = self._assess_confidence(answer_text, relevant_chunks)
            
            print(f"   ✅ Research completed with retrieval method - Confidence: {confidence}, Citations: {has_citations}")
            
            result = {
                "question": question,
                "answer": answer_text,
                "sources": sources,
                "confidence": confidence,
                "has_citations": has_citations
            }
            
            await self.broadcast_event("research_completed", {
                "question": question,
                "success": True,
                "method": retrieval_method,
                "confidence": confidence,
                "sources_count": len(sources)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Research error: {e}")
            await self.broadcast_event("research_completed", {
                "question": question,
                "success": False,
                "error": str(e)
            })
            return {
                "question": question,
                "answer": f"Research failed: {str(e)}",
                "sources": [],
                "confidence": "low",
                "has_citations": False
            }
        finally:
            # Restore original config
            self.config.retrieval_method = original_method
    
    async def _search_documents_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Fast document search
        
        Provides fast search across all documents using pre-computed indexes
        """
        query = params["query"]
        method = params.get("method", "hybrid")
        max_results = params.get("max_results", 20)
        similarity_threshold = params.get("similarity_threshold", 0.01)
        
        print(f"   🔍 Searching documents with {method} method...")
        
        start_time = time.time()
        
        # Use the optimized document processor's fast search
        all_relevant_chunks = []
        pdf_files = list(self.document_processor.page_chunks.keys())
        
        for pdf_file in pdf_files:
            retriever = self.document_processor.get_retriever(pdf_file, method)
            if not retriever:
                continue
            
            try:
                # Fast retrieval using pre-built indexes
                relevant_chunks = retriever.invoke(query)
                
                # Filter by similarity if using vector methods
                if method in ["vector", "hybrid"]:
                    filtered_chunks = self._filter_chunks_by_similarity(query, relevant_chunks, pdf_file, similarity_threshold)
                    relevant_chunks = filtered_chunks
                
                all_relevant_chunks.extend(relevant_chunks)
                
            except Exception as e:
                logger.error(f"Error searching {pdf_file}: {e}")
                continue
        
        # Limit results
        if max_results and len(all_relevant_chunks) > max_results:
            all_relevant_chunks = all_relevant_chunks[:max_results]
        
        search_time = time.time() - start_time
        
        # Extract metadata
        results = []
        for chunk in all_relevant_chunks:
            results.append({
                "content": chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                "source": chunk.metadata.get('source', 'Unknown'),
                "page": chunk.metadata.get('page', 'Unknown'),
                "similarity_score": chunk.metadata.get('similarity_score', 0.0),
                "chunk_id": chunk.metadata.get('chunk_id', 'Unknown')
            })
        
        print(f"   ⚡ Search completed in {search_time:.3f}s - Found {len(results)} results")
        
        return {
            "query": query,
            "method": method,
            "results": results,
            "total_results": len(results),
            "search_time": search_time,
            "sources": list(set([r["source"] for r in results]))
        }
    
    async def _extract_information_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Extract specific information from documents
        
        Targeted information extraction with configurable methods
        """
        extraction_query = params["extraction_query"]
        document_filter = params.get("document_filter", [])
        extraction_method = params.get("extraction_method", "chunks")
        
        print(f"   📊 Extracting information using {extraction_method} method...")
        
        if extraction_method == "full_document":
            # Extract from entire documents
            return await self._extract_from_full_documents(extraction_query, document_filter)
        
        elif extraction_method == "pdf_slices":
            # First find relevant chunks, then use PDF slices
            relevant_chunks = await self._find_relevant_chunks_optimized(extraction_query)
            if document_filter:
                relevant_chunks = [c for c in relevant_chunks if c.metadata.get('source') in document_filter]
            
            if not relevant_chunks:
                return {
                    "query": extraction_query,
                    "method": extraction_method,
                    "extracted_info": "No relevant information found",
                    "sources": [],
                    "confidence": "low"
                }
            
            extracted_text = await self._query_with_pdf_slices_and_check(extraction_query, relevant_chunks)
            
        else:  # chunks method
            # Extract from text chunks
            relevant_chunks = await self._find_relevant_chunks_optimized(extraction_query)
            if document_filter:
                relevant_chunks = [c for c in relevant_chunks if c.metadata.get('source') in document_filter]
            
            if not relevant_chunks:
                return {
                    "query": extraction_query,
                    "method": extraction_method,
                    "extracted_info": "No relevant information found",
                    "sources": [],
                    "confidence": "low"
                }
            
            extracted_text = await self._query_with_chunks_and_check(extraction_query, relevant_chunks)
        
        sources = self._extract_sources_from_chunks(relevant_chunks) if 'relevant_chunks' in locals() else []
        confidence = self._assess_confidence(extracted_text, relevant_chunks) if 'relevant_chunks' in locals() else "medium"
        
        return {
            "query": extraction_query,
            "method": extraction_method,
            "extracted_info": extracted_text,
            "sources": sources,
            "confidence": confidence,
            "has_citations": self._has_source_citations(extracted_text)
        }
    
    async def _analyze_document_quality_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Analyze document quality and relevance
        
        Assess quality metrics for available documents
        """
        topic_keywords = params["topic_keywords"]
        quality_criteria = params.get("quality_criteria", ["completeness", "clarity", "relevance", "citations"])
        
        print(f"   📈 Analyzing document quality for {len(topic_keywords)} keywords...")
        
        document_scores = {}
        
        for pdf_file in self.document_processor.page_chunks.keys():
            scores = {}
            
            # Relevance score based on keyword matches
            if "relevance" in quality_criteria:
                relevance_score = await self._calculate_relevance_score(pdf_file, topic_keywords)
                scores["relevance"] = relevance_score
            
            # Completeness score based on chunk count and coverage
            if "completeness" in quality_criteria:
                completeness_score = self._calculate_completeness_score(pdf_file)
                scores["completeness"] = completeness_score
            
            # Clarity score based on text quality
            if "clarity" in quality_criteria:
                clarity_score = self._calculate_clarity_score(pdf_file)
                scores["clarity"] = clarity_score
            
            # Citation score based on reference patterns
            if "citations" in quality_criteria:
                citation_score = self._calculate_citation_score(pdf_file)
                scores["citations"] = citation_score
            
            # Overall score
            overall_score = sum(scores.values()) / len(scores) if scores else 0.0
            scores["overall"] = overall_score
            
            document_scores[pdf_file] = scores
        
        # Rank documents by overall score
        ranked_documents = sorted(document_scores.items(), key=lambda x: x[1]["overall"], reverse=True)
        
        return {
            "topic_keywords": topic_keywords,
            "quality_criteria": quality_criteria,
            "document_scores": document_scores,
            "ranked_documents": [{"document": doc, "score": scores["overall"]} for doc, scores in ranked_documents],
            "best_document": ranked_documents[0][0] if ranked_documents else None,
            "average_quality": sum(scores["overall"] for scores in document_scores.values()) / len(document_scores) if document_scores else 0.0
        }
    
    async def _get_document_metadata_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MCP Tool: Get document metadata and statistics
        
        Retrieve comprehensive metadata about available documents
        """
        include_statistics = params.get("include_statistics", True)
        include_chunk_info = params.get("include_chunk_info", False)
        document_filter = params.get("document_filter", [])
        
        print(f"   📋 Retrieving document metadata...")
        
        metadata = {
            "total_documents": 0,
            "documents": {},
            "system_info": {
                "cache_path": self.config.cache_path,
                "data_path": self.config.data_path,
                "retrieval_method": self.config.retrieval_method,
                "use_pdf_slices": self.config.use_pdf_slices
            }
        }
        
        documents_to_process = self.document_processor.page_chunks.keys()
        if document_filter:
            documents_to_process = [doc for doc in documents_to_process if doc in document_filter]
        
        for pdf_file in documents_to_process:
            doc_info = {
                "filename": pdf_file,
                "file_path": os.path.join(self.config.data_path, pdf_file),
                "exists": os.path.exists(os.path.join(self.config.data_path, pdf_file))
            }
            
            if doc_info["exists"]:
                # File size
                file_size = os.path.getsize(os.path.join(self.config.data_path, pdf_file))
                doc_info["file_size_mb"] = file_size / (1024 * 1024)
                
                # Page information
                if pdf_file in self.document_processor.page_chunks:
                    doc_info["total_pages"] = len(self.document_processor.page_chunks[pdf_file])
                
                # Chunk information
                if include_chunk_info and pdf_file in self.document_processor.text_chunks:
                    chunks = self.document_processor.text_chunks[pdf_file]
                    doc_info["text_chunks"] = len(chunks)
                    doc_info["avg_chunk_size"] = sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0
                    doc_info["has_embeddings"] = any(chunk.embedding is not None for chunk in chunks)
                
                # Statistics
                if include_statistics:
                    doc_info["statistics"] = self._calculate_document_statistics(pdf_file)
            
            metadata["documents"][pdf_file] = doc_info
            metadata["total_documents"] += 1
        
        # System-wide statistics
        if include_statistics:
            total_pages = sum(doc.get("total_pages", 0) for doc in metadata["documents"].values())
            total_chunks = sum(doc.get("text_chunks", 0) for doc in metadata["documents"].values())
            total_size_mb = sum(doc.get("file_size_mb", 0) for doc in metadata["documents"].values())
            
            metadata["system_statistics"] = {
                "total_pages": total_pages,
                "total_chunks": total_chunks,
                "total_size_mb": total_size_mb,
                "avg_pages_per_doc": total_pages / metadata["total_documents"] if metadata["total_documents"] > 0 else 0,
                "optimization_enabled": True,
                "cache_status": "active"
            }
        
        return metadata
    
    # Helper methods (same logic as original but adapted for async)
    
    async def _find_relevant_chunks_optimized(self, question: str, max_chunks: int = None):
        """Find relevant chunks using pre-computed data - FAST (same as original)"""
        all_relevant_chunks = []
        pdf_files = list(self.document_processor.page_chunks.keys())
        for pdf_file in pdf_files:
            if self.config.retrieval_method == "direct":
                continue
            
            retriever = self.document_processor.get_retriever(pdf_file, self.config.retrieval_method)

            if not retriever:
                continue
            
            try:
                start_time = time.time()
                relevant_chunks = retriever.invoke(question)
                retrieval_time = time.time() - start_time
                
                # Filter by similarity if needed
                if self.config.retrieval_method in ["vector", "hybrid"]:
                    filtered_chunks = self._filter_chunks_by_similarity(question, relevant_chunks, pdf_file)
                    relevant_chunks = filtered_chunks
                
                # Add metadata
                for chunk in relevant_chunks:
                    if 'file_path' not in chunk.metadata:
                        chunk.metadata['file_path'] = os.path.join(self.config.data_path, pdf_file)
                
                all_relevant_chunks.extend(relevant_chunks)
                
            except Exception as e:
                logger.error(f"Error retrieving from {pdf_file}: {e}")
                continue
        
        return all_relevant_chunks
    
    def _filter_chunks_by_similarity(self, question: str, chunks, pdf_file: str, threshold: float = None):
        """Filter chunks by similarity threshold (same as original)"""
        if threshold is None:
            threshold = self.config.similarity_threshold
            
        if not self.document_processor.embeddings_model:
            return chunks
        
        try:
            query_embedding = self.document_processor.embeddings_model.embed_query(question)
            query_embedding = np.array(query_embedding)
            
            filtered_chunks = []
            
            for chunk in chunks:
                chunk_text = chunk.page_content
                similarity_score = 0.0
                
                # Find in precomputed chunks
                if pdf_file in self.document_processor.text_chunks:
                    for precomputed_chunk in self.document_processor.text_chunks[pdf_file]:
                        if precomputed_chunk.content == chunk_text and precomputed_chunk.embedding is not None:
                            similarity_score = np.dot(query_embedding, precomputed_chunk.embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(precomputed_chunk.embedding)
                            )
                            break
                
                if similarity_score >= threshold:
                    chunk.metadata['similarity_score'] = similarity_score
                    filtered_chunks.append(chunk)
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error filtering by similarity: {e}")
            return chunks
    
    async def _query_with_pdf_slices_and_check(self, question: str, chunks):
        """Query with PDF slices and relevance checking (same as original logic)"""
        try:
            if not self.config.genai_client:
                return await self._query_with_chunks_and_check(question, chunks)
            
            # Extract file paths and page numbers
            pdf_slices = []
            for chunk in chunks:
                slice_info = {"page": 1}
                
                if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                    metadata = chunk.metadata
                    
                    # Get file path
                    file_path = None
                    for key in ['file_path', 'source', 'path']:
                        if key in metadata and metadata[key]:
                            file_path = metadata[key]
                            break
                    
                    if file_path:
                        if os.path.isabs(file_path):
                            final_file_path = file_path
                        elif file_path.startswith('./') or '/' in file_path:
                            final_file_path = file_path
                        else:
                            final_file_path = os.path.join(self.config.data_path, file_path)
                        slice_info["file_path"] = final_file_path
                    
                    # Get page number
                    page_num = None
                    for key in ['page', 'page_number', 'page_num']:
                        if key in metadata and metadata[key]:
                            try:
                                page_num = int(metadata[key])
                                break
                            except (ValueError, TypeError):
                                pass
                    
                    if page_num:
                        slice_info["page"] = page_num
                
                if "file_path" in slice_info and os.path.exists(slice_info["file_path"]):
                    pdf_slices.append(slice_info)
            
            if not pdf_slices:
                return await self._query_with_chunks_and_check(question, chunks)
            
            # Group slices by file and add page buffers
            files_to_pages = {}
            for s in pdf_slices:
                file_path = s['file_path']
                page = int(s['page'])
                
                if file_path not in files_to_pages:
                    files_to_pages[file_path] = set()
                
                for offset in range(-self.config.page_buffer, self.config.page_buffer + 1):
                    buffered_page = page + offset
                    if buffered_page > 0:
                        files_to_pages[file_path].add(buffered_page)
            
            # Create temporary PDF with relevant pages
            output_pdf = fitz.open()
            added_pages = {}
            total_pages_added = 0
            
            for file_path, pages in files_to_pages.items():
                try:
                    doc = fitz.open(file_path)
                    total_pages = len(doc)
                    file_name = os.path.basename(file_path)
                    
                    if file_name not in added_pages:
                        added_pages[file_name] = []
                    
                    valid_pages = []
                    for p in sorted(pages):
                        if 1 <= p <= total_pages:
                            valid_pages.append(p - 1)  # Convert to 0-indexed
                    
                    for page_num in valid_pages:
                        try:
                            if page_num < 0 or page_num >= total_pages:
                                continue
                            
                            output_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                            added_pages[file_name].append(page_num + 1)
                            total_pages_added += 1
                            
                        except Exception:
                            continue
                    
                    doc.close()
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            if total_pages_added == 0:
                output_pdf.close()
                return await self._query_with_chunks_and_check(question, chunks)
            
            # Save temporary PDF and query Gemini
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"temp_slice_{int(time.time())}.pdf")
            output_pdf.save(temp_pdf_path)
            output_pdf.close()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            with open(temp_pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            enhanced_question = f"""
            {question}
            
            IMPORTANT: If the document does not contain sufficient information to answer this question, please explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."
            
            Always include specific page numbers in your citations.
            """
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    enhanced_question
                ]
            )
            
            result = response.text
            
            # Create page range citations
            citations = []
            for file_name, pages in added_pages.items():
                if not pages:
                    continue
                    
                pages = sorted(list(set(pages)))
                ranges = []
                
                if len(pages) > 0:
                    range_start = pages[0]
                    prev_page = pages[0]
                    
                    for page in pages[1:]:
                        if page > prev_page + 1:
                            if range_start == prev_page:
                                ranges.append(f"{range_start}")
                            else:
                                ranges.append(f"{range_start}-{prev_page}")
                            range_start = page
                        prev_page = page
                    
                    if range_start == prev_page:
                        ranges.append(f"{range_start}")
                    else:
                        ranges.append(f"{range_start}-{prev_page}")
                
                citations.append(f"pp. {', '.join(ranges)} ({file_name})")
            
            if citations:
                result += f"\n\nSources: {'; '.join(citations)}"
            
            # Clean up
            try:
                os.remove(temp_pdf_path)
            except Exception:
                pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PDF slice query: {e}")
            return await self._query_with_chunks_and_check(question, chunks)
    
    async def _query_with_chunks_and_check(self, question: str, chunks):
        """Query with text chunks and explicit relevance checking (same as original)"""
        # Prepare context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}, Page: {chunk.metadata.get('page', 'Unknown')}"
            chunk_text = chunk.page_content[:1500]
            context_parts.append(f"[Chunk {i+1}] {source_info}\n{chunk_text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""
        You are analyzing corporate governance documents to answer a specific question.
        
        QUESTION: {question}
        
        DOCUMENT EXCERPTS:
        {context}
        
        Instructions:
        1. Answer the question based ONLY on the provided document excerpts
        2. ALWAYS include specific source citations (page numbers and document names)
        3. If the excerpts do not contain sufficient information to answer the question, explicitly state: "INSUFFICIENT INFORMATION: The provided excerpts do not contain enough details to fully answer this question."
        4. Be precise and factual
        5. Format citations as: "According to [document name], page [number]..."
        
        IMPORTANT: If you cannot find relevant information in the excerpts, clearly state that the information is not available rather than making assumptions.
        
        Provide a comprehensive answer with proper source citations, or clearly indicate if information is insufficient.
        """
        
        try:
            if hasattr(self.llm, 'generate_content'):
                response = self.llm.generate_content(prompt)
                result = response.text
            else:
                result = self.llm.invoke(prompt)
            
            return result
                
        except Exception as e:
            logger.error(f"LLM query error: {e}")
            return f"Error querying documents: {str(e)}"
    
    def _should_fallback_to_direct(self, answer_text: str, question: str) -> bool:
        """Check if should fallback to direct method (same as original)"""
        if not self.config.auto_fallback_to_direct:
            return False
        
        answer_lower = answer_text.lower()
        
        if "INSUFFICIENT INFORMATION:" in answer_text:
            return True
        
        insufficient_indicators = 0
        for keyword in self.config.fallback_keywords:
            if keyword.lower() in answer_lower:
                insufficient_indicators += 1
        
        if insufficient_indicators >= 2:
            return True
        
        if len(answer_text.strip()) < 100 and insufficient_indicators >= 1:
            return True
        
        return False
    
    async def _fallback_to_direct_method(self, question: str, reason: str, relevant_chunks=None):
        """Fallback to direct method with progressive escalation (exact same logic as original)"""
        
        print(f"   🔄 FALLING BACK TO DIRECT METHOD")
        print(f"   📋 Reason: {reason}")
        
        # Identify and rank documents by relevance (exact same logic as original)
        if relevant_chunks:
            # Use documents identified by retrieval with relevance ranking
            relevant_docs = {}
            for chunk in relevant_chunks:
                doc_name = chunk.metadata.get('source', 'Unknown')
                if doc_name not in relevant_docs:
                    relevant_docs[doc_name] = 0
                relevant_docs[doc_name] += 1
            
            # Sort by relevance (chunk count)
            sorted_docs = sorted(relevant_docs.items(), key=lambda x: x[1], reverse=True)
            
            print(f"   📊 Document relevance ranking:")
            for doc_name, chunk_count in sorted_docs:
                print(f"     • {doc_name}: {chunk_count} relevant chunks")
        else:
            # Use all available documents if no retrieval info
            pdf_files = [f for f in os.listdir(self.config.data_path) if f.endswith('.pdf')]
            sorted_docs = [(doc_name, 0) for doc_name in pdf_files]
            
            print(f"   📂 Using all available documents (no retrieval ranking):")
            for doc_name, _ in sorted_docs:
                print(f"     • {doc_name}")
        
        # Progressive escalation: try one document at a time (exact same logic as original)
        print(f"   🎯 Starting progressive escalation strategy...")
        
        combined_context = ""
        processed_docs = []
        
        for attempt, (doc_name, chunk_count) in enumerate(sorted_docs[:self.config.max_direct_documents], 1):
            doc_path = os.path.join(self.config.data_path, doc_name)
            
            if not os.path.exists(doc_path):
                print(f"   ❌ Document not found: {doc_name}")
                continue
            
            # Check file size (exact same logic as original)
            file_size_mb = os.path.getsize(doc_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ {doc_name} too large ({file_size_mb:.1f} MB), skipping")
                continue
            
            try:
                print(f"   📖 Attempt {attempt}: Processing {doc_name} ({chunk_count} relevant chunks, {file_size_mb:.1f} MB)")
                
                # Enhanced question for progressive attempts (exact same logic as original)
                if attempt == 1:
                    enhanced_question = f"""
{question}

IMPORTANT: Please provide a complete and comprehensive answer. If this document does not contain sufficient information to fully answer the question, explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."

Include specific page numbers in your citations.
"""
                else:
                    enhanced_question = f"""
{question}

CONTEXT: Previous document(s) provided some information but it was insufficient. This is attempt #{attempt} to find complete information.

IMPORTANT: Please provide a complete and comprehensive answer based on this document. If this document ALSO does not contain sufficient information to fully answer the question, explicitly state: "INSUFFICIENT INFORMATION: This document does not contain enough details to fully answer this question."

Include specific page numbers in your citations.
"""
                
                answer = await self._query_entire_document_with_enhanced_question(doc_path, enhanced_question)
                
                if answer:
                    processed_docs.append(doc_name)
                    
                    # Check if this attempt was sufficient (exact same logic as original)
                    if not self._should_fallback_to_direct(answer, question):
                        print(f"   ✅ SUCCESS on attempt {attempt}! {doc_name} provided sufficient information")
                        
                        # Combine with any previous context if this wasn't the first attempt
                        if combined_context:
                            final_answer = f"{combined_context}\n\n=== ADDITIONAL INFORMATION FROM {doc_name} ===\n{answer}"
                        else:
                            final_answer = f"=== FROM {doc_name} ===\n{answer}"
                        
                        return {
                            "question": question,
                            "answer": final_answer.strip(),
                            "sources": processed_docs,
                            "confidence": "high" if attempt == 1 else "medium",
                            "has_citations": self._has_source_citations(answer)
                        }
                    else:
                        print(f"   ⚠️ Attempt {attempt} insufficient, escalating to next document...")
                        # Add this document's info to context for next attempt
                        combined_context += f"\n\n=== FROM {doc_name} ===\n{answer}"
                else:
                    print(f"   ⚠️ No response from {doc_name}")
                    
            except Exception as e:
                logger.error(f"Error processing {doc_name} with direct method: {e}")
                print(f"   ❌ Error processing {doc_name}: {e}")
                continue
        
        # If we get here, all attempts failed (exact same logic as original)
        print(f"   ❌ Progressive escalation failed after {len(processed_docs)} attempts")
        
        if combined_context:
            # Return what we found, even if insufficient
            return {
                "question": question,
                "answer": f"Partial information found across {len(processed_docs)} documents:\n{combined_context}",
                "sources": processed_docs,
                "confidence": "low",
                "has_citations": self._has_source_citations(combined_context)
            }
        else:
            return {
                "question": question,
                "answer": f"Could not find relevant information using progressive direct method. Reason: {reason}",
                "sources": [],
                "confidence": "low",
                "has_citations": False
            }
    
    async def _query_entire_document_with_enhanced_question(self, document_path: str, enhanced_question: str) -> str:
        """Query entire document with enhanced question for progressive escalation (exact same as original)"""
        try:
            if not self.config.genai_client:
                return None
            
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                print(f"   ⚠️ Document too large: {file_size_mb:.2f} MB > {self.config.max_pdf_size_mb} MB")
                return None
            
            with open(document_path, 'rb') as f:
                pdf_bytes = f.read()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            print_llm_interaction("Progressive Direct Method Query", enhanced_question, "", truncate=True)
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    enhanced_question
                ]
            )
            
            result = response.text
            print_llm_interaction("Progressive Direct Method Query", "", result, truncate=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying entire document: {e}")
            return None
    
    async def _query_entire_document(self, document_path: str, enhanced_question: str) -> str:
        """Query entire document with enhanced question (same as original)"""
        try:
            if not self.config.genai_client:
                return None
            
            file_size_mb = os.path.getsize(document_path) / (1024 * 1024)
            if file_size_mb > self.config.max_pdf_size_mb:
                return None
            
            with open(document_path, 'rb') as f:
                pdf_bytes = f.read()
            
            model = self.config.genai_client.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(
                contents=[
                    {"mime_type": "application/pdf", 
                     "data": base64.b64encode(pdf_bytes).decode('utf-8')},
                    enhanced_question
                ]
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error querying entire document: {e}")
            return None
    
    def _extract_sources_from_chunks(self, chunks):
        """Extract source information from chunks (same as original)"""
        sources = []
        for chunk in chunks:
            metadata = chunk.metadata
            source_info = f"Page {metadata.get('page', 'Unknown')} ({metadata.get('source', 'Unknown')})"
            if source_info not in sources:
                sources.append(source_info)
        return sources
    
    def _has_source_citations(self, answer_text: str) -> bool:
        """Check if answer contains source citations (same as original)"""
        citation_patterns = ['page', 'source:', 'according to', 'document', 'pp.', 'from']
        answer_lower = answer_text.lower()
        return any(pattern in answer_lower for pattern in citation_patterns)
    
    def _assess_confidence(self, answer_text: str, chunks) -> str:
        """Assess confidence based on answer quality (same as original)"""
        if len(chunks) >= 5 and len(answer_text) > 200 and self._has_source_citations(answer_text):
            return "high"
        elif len(chunks) >= 3 and len(answer_text) > 100:
            return "medium"
        else:
            return "low"
    
    # Additional helper methods for new tools
    
    async def _extract_from_full_documents(self, extraction_query: str, document_filter: List[str]) -> Dict[str, Any]:
        """Extract information from full documents"""
        results = []
        sources = []
        
        documents_to_process = list(self.document_processor.page_chunks.keys())
        if document_filter:
            documents_to_process = [doc for doc in documents_to_process if doc in document_filter]
        
        for doc_name in documents_to_process:
            doc_path = os.path.join(self.config.data_path, doc_name)
            if not os.path.exists(doc_path):
                continue
            
            try:
                extracted_text = await self._query_entire_document(doc_path, extraction_query)
                if extracted_text:
                    results.append(f"=== FROM {doc_name} ===\n{extracted_text}")
                    sources.append(doc_name)
            except Exception as e:
                logger.error(f"Error extracting from {doc_name}: {e}")
                continue
        
        combined_result = "\n\n".join(results) if results else "No relevant information found"
        
        return {
            "query": extraction_query,
            "method": "full_document",
            "extracted_info": combined_result,
            "sources": sources,
            "confidence": "high" if len(sources) > 1 else "medium" if len(sources) == 1 else "low"
        }
    
    async def _calculate_relevance_score(self, pdf_file: str, topic_keywords: List[str]) -> float:
        """Calculate relevance score for a document based on topic keywords"""
        if pdf_file not in self.document_processor.text_chunks:
            return 0.0
        
        chunks = self.document_processor.text_chunks[pdf_file]
        total_content = " ".join([chunk.content for chunk in chunks]).lower()
        
        keyword_matches = 0
        for keyword in topic_keywords:
            if keyword.lower() in total_content:
                keyword_matches += 1
        
        return keyword_matches / len(topic_keywords) if topic_keywords else 0.0
    
    def _calculate_completeness_score(self, pdf_file: str) -> float:
        """Calculate completeness score based on document structure"""
        if pdf_file not in self.document_processor.page_chunks:
            return 0.0
        
        page_chunks = self.document_processor.page_chunks[pdf_file]
        text_chunks = self.document_processor.text_chunks.get(pdf_file, [])
        
        # Score based on page count, chunk distribution, etc.
        page_count = len(page_chunks)
        chunk_count = len(text_chunks)
        
        # Normalize scores (simple heuristic)
        page_score = min(page_count / 50, 1.0)  # Assume 50 pages is "complete"
        chunk_score = min(chunk_count / 100, 1.0)  # Assume 100 chunks is "complete"
        
        return (page_score + chunk_score) / 2
    
    def _calculate_clarity_score(self, pdf_file: str) -> float:
        """Calculate clarity score based on text quality"""
        if pdf_file not in self.document_processor.text_chunks:
            return 0.0
        
        chunks = self.document_processor.text_chunks[pdf_file]
        
        # Simple heuristics for text clarity
        total_length = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_length = total_length / len(chunks) if chunks else 0
        
        # Score based on average chunk length (neither too short nor too long)
        if 500 <= avg_chunk_length <= 1500:
            return 1.0
        elif 200 <= avg_chunk_length <= 2000:
            return 0.7
        else:
            return 0.4
    
    def _calculate_citation_score(self, pdf_file: str) -> float:
        """Calculate citation score based on reference patterns"""
        if pdf_file not in self.document_processor.text_chunks:
            return 0.0
        
        chunks = self.document_processor.text_chunks[pdf_file]
        total_content = " ".join([chunk.content for chunk in chunks]).lower()
        
        # Look for citation patterns
        citation_patterns = ['page', 'section', 'clause', 'paragraph', 'according to', 'as per']
        citation_count = sum(total_content.count(pattern) for pattern in citation_patterns)
        
        # Normalize based on document length
        words_count = len(total_content.split())
        citation_density = citation_count / words_count if words_count > 0 else 0
        
        return min(citation_density * 1000, 1.0)  # Scale and cap at 1.0
    
    def _calculate_document_statistics(self, pdf_file: str) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a document"""
        stats = {
            "page_chunks": 0,
            "text_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0,
            "has_embeddings": False
        }
        
        if pdf_file in self.document_processor.page_chunks:
            stats["page_chunks"] = len(self.document_processor.page_chunks[pdf_file])
        
        if pdf_file in self.document_processor.text_chunks:
            chunks = self.document_processor.text_chunks[pdf_file]
            stats["text_chunks"] = len(chunks)
            stats["total_characters"] = sum(len(chunk.content) for chunk in chunks)
            stats["avg_chunk_size"] = stats["total_characters"] / len(chunks) if chunks else 0
            stats["has_embeddings"] = any(chunk.embedding is not None for chunk in chunks)
        
        return stats
    
    async def _handle_broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Handle broadcast events from other agents"""
        if event_type == "question_generated":
            print(f"   📢 New question generated: {data.get('question', 'Unknown')[:100]}...")
        
        elif event_type == "evidence_gap_identified":
            print(f"   📢 Evidence gap identified: {data.get('gap_description', 'Unknown')}")
        
        elif event_type == "evaluation_started":
            print(f"   📢 Evaluation started, research agent ready")

# Backward compatibility wrapper
class OptimizedResearchAgentWrapper:
    """
    Wrapper to maintain backward compatibility with original interface
    """
    
    def __init__(self, mcp_agent: MCPResearchAgent):
        self.mcp_agent = mcp_agent
        self.current_model = mcp_agent.current_model
        self.document_processor = mcp_agent.document_processor
    
    async def research_question(self, question: str) -> Answer:
        """Original interface method - calls MCP tool internally"""
        result = await self.mcp_agent.call_tool("research_question", {
            "question": question
        })
        
        return Answer(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            has_citations=result["has_citations"]
        )

# Factory function for creating MCP Research Agent
def create_mcp_research_agent(config: OptimizedConfig, message_bus) -> MCPResearchAgent:
    """Factory function to create MCP Research Agent"""
    return MCPResearchAgent(config, message_bus)