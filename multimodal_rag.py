import os
import torch
from typing import List, Union, Dict, Any, Optional
from pathlib import Path
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


class RAGSystem:
    """
    A multimodal RAG system that uses ColPali for document retrieval and Qwen2-VL for question answering.
    """
    
    def __init__(
        self,
        colpali_model: str = "vidore/colpali-v1.2",
        vlm_model: str = "Qwen/Qwen2-VL-7B-Instruct",
        index_path: str = "./index",
        use_gpu: bool = True,
        low_memory: bool = False
    ):
        """
        Initialize the RAG system.
        
        Args:
            colpali_model: The ColPali model to use for document retrieval.
            vlm_model: The Vision-Language Model to use for question answering.
            index_path: Path to store the document index.
            use_gpu: Whether to use GPU for inference.
            low_memory: Whether to use low memory mode (slower but uses less VRAM).
        """
        self.colpali_model = colpali_model
        self.vlm_model = vlm_model
        self.index_path = index_path
        self.use_gpu = use_gpu
        self.low_memory = low_memory
        
        # Load the document retrieval model
        print(f"Loading document retrieval model: {colpali_model}")
        self.retrieval_model = RAGMultiModalModel.from_pretrained(colpali_model)
        
        # Set the device
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load the Vision-Language Model
        print(f"Loading VLM model: {vlm_model}")
        
        # Determine dtype based on available hardware
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Load model with appropriate settings
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if not low_memory else None,
            device_map=self.device if not low_memory else None,
            low_cpu_mem_usage=True
        )
        
        if low_memory:
            self.vlm = self.vlm.to(self.device)
        
        self.vlm.eval()
        
        # Load the processor
        min_pixels = 224 * 224  # Minimum image size
        max_pixels = 1024 * 1024  # Maximum image size
        self.processor = Qwen2VLProcessor.from_pretrained(
            vlm_model,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # Document cache for efficient retrieval
        self.doc_images = {}
        self.index_name = None
        
    def index_documents(
        self, 
        folder_path: Optional[str] = None, 
        file_paths: Optional[List[str]] = None,
        index_name: Optional[str] = None,
        overwrite: bool = False
    ) -> None:
        """
        Index documents for retrieval.
        
        Args:
            folder_path: Path to a folder containing PDF documents.
            file_paths: List of paths to individual PDF documents.
            index_name: Name of the index to create.
            overwrite: Whether to overwrite an existing index.
        """
        if folder_path is None and file_paths is None:
            raise ValueError("Either folder_path or file_paths must be provided.")
        
        # Generate a unique index name if not provided
        if index_name is None:
            # Use folder name or first file name as index name
            if folder_path:
                index_name = os.path.basename(os.path.normpath(folder_path))
            else:
                index_name = os.path.basename(file_paths[0]).split('.')[0]
            
        self.index_name = index_name
        
        # Create the index
        if folder_path:
            print(f"Indexing documents from folder: {folder_path}")
            self.retrieval_model.index(
                input_path=folder_path,
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=overwrite
            )
        else:
            # Create a temporary folder for indexing if individual files are provided
            temp_folder = os.path.join(self.index_path, "temp_index")
            os.makedirs(temp_folder, exist_ok=True)
            
            # Copy files to the temporary folder
            import shutil
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                dest_path = os.path.join(temp_folder, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
            
            print(f"Indexing {len(file_paths)} documents")
            self.retrieval_model.index(
                input_path=temp_folder,
                index_name=index_name,
                store_collection_with_index=True,
                overwrite=overwrite
            )
            
            # Clean up temporary folder
            shutil.rmtree(temp_folder)
        
        print(f"Indexing complete. Index name: {index_name}")
    
    def _get_document_images(self, doc_id: int) -> List:
        """
        Get the images for a document.
        
        Args:
            doc_id: The document ID.
            
        Returns:
            A list of images for the document.
        """
        if doc_id in self.doc_images:
            return self.doc_images[doc_id]
        
        # Get the original document path
        doc_info = self.retrieval_model.get_document_path(doc_id)
        
        if doc_info is None:
            raise ValueError(f"Document with ID {doc_id} not found.")
        
        # Convert PDF to images
        images = convert_from_path(doc_info["path"])
        self.doc_images[doc_id] = images
        
        return images
    
    def answer_question(
        self, 
        question: str, 
        top_k: int = 3, 
        max_tokens: int = 500
    ) -> str:
        """
        Answer a question based on the indexed documents.
        
        Args:
            question: The question to answer.
            top_k: Number of relevant document pages to retrieve.
            max_tokens: Maximum number of tokens in the answer.
            
        Returns:
            The answer to the question.
        """
        if self.index_name is None:
            raise ValueError("No documents have been indexed. Call index_documents() first.")
        
        # Search for relevant document pages
        print(f"Searching for relevant pages for question: {question}")
        results = self.retrieval_model.search(question, k=top_k)
        
        if not results:
            return "No relevant documents found. Please try a different question or index some documents first."
        
        # Get the images for the retrieved pages
        retrieved_images = []
        for result in results:
            doc_id = result["doc_id"]
            page_num = result["page_num"]
            
            # Get all images for the document
            doc_images = self._get_document_images(doc_id)
            
            # Page numbers are 1-indexed in search results
            image_index = page_num - 1
            
            if 0 <= image_index < len(doc_images):
                retrieved_images.append(doc_images[image_index])
            else:
                print(f"Warning: Page {page_num} not found in document {doc_id}")
        
        if not retrieved_images:
            return "Failed to retrieve document images. Please check your document index."
        
        # Prepare the chat template
        chat_template = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image} for image in retrieved_images
                ] + [{"type": "text", "text": question}]
            }
        ]
        
        # Prepare inputs for the VLM
        text = self.processor.apply_chat_template(
            chat_template, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(chat_template)
        
        inputs = self.processor(
            text=[text], 
            images=image_inputs, 
            padding=True, 
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # Generate the answer
        with torch.no_grad():
            generated_ids = self.vlm.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=False
            )
        
        # Process the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Index a folder of PDF documents
    rag_system.index_documents(folder_path="./sample_pdfs")
    
    # Answer a question
    answer = rag_system.answer_question("What does Figure 2 show?")
    print(f"Answer: {answer}")
