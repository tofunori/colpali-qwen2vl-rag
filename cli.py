#!/usr/bin/env python3
"""
Command-line interface for the ColPali + Qwen2-VL Multimodal RAG System.
"""

import os
import sys
import argparse
from typing import List, Optional
from pathlib import Path
from multimodal_rag import RAGSystem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ColPali + Qwen2-VL Multimodal RAG System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "--folder", 
        type=str, 
        help="Path to folder containing PDFs"
    )
    index_parser.add_argument(
        "--files", 
        type=str, 
        nargs="+", 
        help="Paths to PDF files"
    )
    index_parser.add_argument(
        "--index-name", 
        type=str, 
        help="Name for the index"
    )
    index_parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite existing index"
    )
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument(
        "question",
        type=str,
        help="Question to ask"
    )
    ask_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of pages to retrieve"
    )
    ask_parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens in the answer"
    )
    
    # Model configuration options
    for p in [index_parser, ask_parser]:
        p.add_argument(
            "--colpali-model", 
            type=str, 
            default="vidore/colpali-v1.2",
            help="ColPali model to use"
        )
        p.add_argument(
            "--vlm-model", 
            type=str, 
            default="Qwen/Qwen2-VL-7B-Instruct",
            help="Vision Language Model to use"
        )
        p.add_argument(
            "--index-path", 
            type=str, 
            default="./index",
            help="Path to store index"
        )
        p.add_argument(
            "--cpu", 
            action="store_true",
            help="Use CPU instead of GPU"
        )
        p.add_argument(
            "--low-memory", 
            action="store_true",
            help="Use low memory mode"
        )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "index" and args.folder is None and args.files is None:
        index_parser.error("Either --folder or --files must be specified")
    
    return args


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = RAGSystem(
        colpali_model=args.colpali_model,
        vlm_model=args.vlm_model,
        index_path=args.index_path,
        use_gpu=not args.cpu,
        low_memory=args.low_memory
    )
    
    if args.command == "index":
        # Index documents
        rag_system.index_documents(
            folder_path=args.folder,
            file_paths=args.files,
            index_name=args.index_name,
            overwrite=args.overwrite
        )
        print("Indexing complete!")
        
    elif args.command == "ask":
        # Answer a question
        print(f"Question: {args.question}")
        print("Thinking...")
        answer = rag_system.answer_question(
            question=args.question,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
        print("\nAnswer:")
        print(f"{answer}")


if __name__ == "__main__":
    main()
