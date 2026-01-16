from langchain_chroma import Chroma
from typing_extensions import override

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.helpers.data import docs_to_data
from lfx.inputs.inputs import BoolInput, HandleInput, IntInput, MessageTextInput
from lfx.io import Output, QueryInput
from lfx.schema.data import Data
from lfx.schema.message import Message


class ChromaSearchAgentComponent(LCVectorStoreComponent):
    """Chroma DB Search Agent - Simple component to search a Chroma database."""

    display_name: str = "Chroma Search Agent"
    description: str = "Search a Chroma database using a persist directory path and user query."
    name = "ChromaSearchAgent"
    icon = "Chroma"

    inputs = [
        MessageTextInput(
            name="persist_directory",
            display_name="Persist Directory",
            info="Path to the Chroma database directory (e.g., './chroma_db' or '/path/to/chroma_db'). Can accept text input or connection from another node.",
            required=True,
            show=True,
            value="",
        ),
        QueryInput(
            name="search_query",
            display_name="Search Query",
            info="Enter a query to run a similarity search.",
            placeholder="Enter a query...",
            tool_mode=True,
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            info="Embedding model to use for vector search. Required for searching.",
            required=True,
            show=True,
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=10,
        ),
        BoolInput(
            name="should_cache_vector_store",
            display_name="Cache Vector Store",
            value=True,
            advanced=True,
            info="If True, the vector store will be cached for the current build of the component.",
        ),
    ]

    outputs = [
        Output(
            display_name="Category",
            name="category",
            method="search_documents",
        ),
    ]

    @override
    @check_cached_vector_store
    def build_vector_store(self) -> Chroma:
        """Builds the Chroma object from the persist directory."""
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            msg = "Could not import Chroma integration package. Please install it with `pip install langchain-chroma`."
            raise ImportError(msg) from e

        # Get persist_directory - handle both string and Message inputs
        persist_dir_value = self.persist_directory
        if isinstance(persist_dir_value, str):
            persist_dir_str = persist_dir_value
        elif hasattr(persist_dir_value, "text"):
            # Handle Message objects
            persist_dir_str = str(persist_dir_value.text)
        elif hasattr(persist_dir_value, "data") and isinstance(persist_dir_value.data, dict):
            # Handle Data objects
            persist_dir_str = str(persist_dir_value.data.get("text", persist_dir_value.data.get("content", str(persist_dir_value))))
        else:
            persist_dir_str = str(persist_dir_value) if persist_dir_value else ""

        # Check persist_directory and expand it if it is a relative path
        persist_directory = self.resolve_path(persist_dir_str) if persist_dir_str else None

        if not persist_directory:
            msg = "Persist directory is required to connect to Chroma database."
            raise ValueError(msg)

        # Build Chroma instance
        # Use default collection name "langflow" if not specified
        collection_name = getattr(self, "collection_name", "langflow")
        chroma = Chroma(
            persist_directory=persist_directory,
            client=None,
            embedding_function=self.embedding,
            collection_name=collection_name,
        )

        # Log connection status
        try:
            collection_info = chroma.get()
            num_docs = len(collection_info.get("ids", []))
            collection_name = getattr(self, "collection_name", "langflow")
            self.log(f"Connected to Chroma DB at '{persist_directory}' (collection: '{collection_name}', documents: {num_docs})")
        except Exception as e:
            self.log(f"Warning: Could not retrieve collection info: {e}")

        return chroma

    @override
    def search_documents(self) -> Message:
        """Search for documents in the Chroma vector store and return category based on search results."""
        # Define the exact category terms to search for
        categories = [
            "demographics",
            "group_info",
            "medical_claim",
            "j_code",
            "medical_savings",
            "pharm_and_phys",
            "pharmacy_claim",
            "opioid_and_benzo",
            "pharmacy_savings",
            "Clarification needed",
        ]
        
        # Use the parent class's search mechanism (same as working component)
        # This ensures compatibility and uses the proven search method
        try:
            # Call parent's search_documents to get results
            # First ensure vector store is built
            if self._cached_vector_store is not None:
                vector_store = self._cached_vector_store
            else:
                vector_store = self.build_vector_store()
                self._cached_vector_store = vector_store

            # Get search_query from parent class inputs
            search_query: str = self.search_query
            if not search_query:
                self.status = "No search query provided."
                return Message(text="Clarification needed")

            self.log(f"Searching Chroma DB with query: '{search_query}'")
            self.log(f"Number of results requested: {self.number_of_results}")

            # Use Chroma's native similarity_search_with_score method
            # This is the most reliable method for Chroma (langchain_chroma wrapper)
            search_results = []
            try:
                if hasattr(vector_store, "similarity_search_with_score"):
                    docs_with_scores = vector_store.similarity_search_with_score(
                        query=search_query,
                        k=self.number_of_results,
                    )
                    
                    self.log(f"Found {len(docs_with_scores)} documents with scores")
                    
                    if not docs_with_scores:
                        self.log("No results found for the query.")
                        self.status = "No results found."
                        return Message(text="Clarification needed")
                    
                    # Extract documents from (doc, score) tuples
                    docs = [doc for doc, score in docs_with_scores]
                    
                    # Log document details for debugging
                    for i, doc in enumerate(docs):
                        page_content = doc.page_content if hasattr(doc, "page_content") else ""
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        self.log(f"Document {i+1}: page_content length={len(page_content)}, preview={page_content[:100]}, metadata={metadata}")
                    
                    # Convert documents to Data objects
                    search_results = docs_to_data(docs)
                    self.log(f"Converted to {len(search_results)} Data objects")
                    
                elif hasattr(vector_store, "similarity_search"):
                    # Fallback: try similarity_search if similarity_search_with_score doesn't exist
                    self.log("similarity_search_with_score not available, trying similarity_search")
                    docs = vector_store.similarity_search(
                        query=search_query,
                        k=self.number_of_results,
                    )
                    
                    self.log(f"Found {len(docs)} documents")
                    
                    if not docs:
                        self.log("No results found for the query.")
                        self.status = "No results found."
                        return Message(text="Clarification needed")
                    
                    # Log document details for debugging
                    for i, doc in enumerate(docs):
                        page_content = doc.page_content if hasattr(doc, "page_content") else ""
                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                        self.log(f"Document {i+1}: page_content length={len(page_content)}, preview={page_content[:100]}, metadata={metadata}")
                    
                    # Convert documents to Data objects
                    search_results = docs_to_data(docs)
                    self.log(f"Converted to {len(search_results)} Data objects")
                else:
                    error_msg = "Vector store does not have similarity_search or similarity_search_with_score methods"
                    self.log(error_msg)
                    self.status = error_msg
                    return Message(text="Clarification needed")
                    
            except Exception as search_error:
                error_msg = f"Error during similarity search: {search_error}"
                self.log(error_msg)
                import traceback
                self.log(f"Search error traceback: {traceback.format_exc()}")
                self.status = error_msg
                return Message(text="Clarification needed")
            
            if not search_results:
                self.log("No results found for the query (search_results is empty).")
                self.status = "No results found."
                return Message(text="Clarification needed")
            
            self.log(f"Have {len(search_results)} search results to process")
            
            # Search through all results for category terms
            # Check both text content and metadata
            combined_text = ""
            for i, result in enumerate(search_results):
                self.log(f"Processing result {i+1}: type={type(result).__name__}")
                
                # Try multiple ways to extract text content
                text_extracted = False
                
                # Method 1: Check for text attribute
                if hasattr(result, "text") and result.text:
                    text_content = str(result.text)
                    combined_text += " " + text_content
                    self.log(f"  Found text attribute: {text_content[:150]}...")
                    text_extracted = True
                
                # Method 2: Check for data dict with text fields
                if hasattr(result, "data") and isinstance(result.data, dict):
                    # Check various text fields in data
                    for key in ["text", "content", "page_content", "data"]:
                        if key in result.data and result.data[key]:
                            text_content = str(result.data[key])
                            combined_text += " " + text_content
                            self.log(f"  Found data[{key}]: {text_content[:150]}...")
                            text_extracted = True
                    
                    # Also check all metadata values
                    if "metadata" in result.data and isinstance(result.data["metadata"], dict):
                        for key, value in result.data["metadata"].items():
                            if value:
                                combined_text += " " + str(value)
                                self.log(f"  Found metadata[{key}]: {str(value)[:150]}...")
                                text_extracted = True
                
                # Method 3: Check for page_content attribute (Document objects)
                if hasattr(result, "page_content") and result.page_content:
                    text_content = str(result.page_content)
                    combined_text += " " + text_content
                    self.log(f"  Found page_content: {text_content[:150]}...")
                    text_extracted = True
                
                # Method 4: Check for metadata attribute directly
                if hasattr(result, "metadata") and isinstance(result.metadata, dict):
                    for key, value in result.metadata.items():
                        if value:
                            combined_text += " " + str(value)
                            self.log(f"  Found result.metadata[{key}]: {str(value)[:150]}...")
                            text_extracted = True
                
                if not text_extracted:
                    self.log(f"  WARNING: No text extracted from result {i+1}. Result attributes: {dir(result)}")
                    # Try to convert entire result to string as last resort
                    combined_text += " " + str(result)
                    self.log(f"  Using string representation: {str(result)[:150]}...")
            
            self.log(f"Combined text length: {len(combined_text)} characters")
            if combined_text.strip():
                self.log(f"Combined text preview (first 500 chars): {combined_text[:500]}")
            else:
                self.log("WARNING: Combined text is empty after processing all results!")
            
            # Normalize text for searching (lowercase)
            combined_text_lower = combined_text.lower()
            self.log(f"Searching for categories in combined text (first 500 chars): {combined_text_lower[:500]}")
            
            # Search for each category term (case-insensitive) and extract queries
            # Check in order, return first match found with its queries
            for category in categories:
                category_lower = category.lower()
                if category_lower in combined_text_lower:
                    self.log(f"Found category '{category}' in search results")
                    
                    # Extract full content chunks from ALL matching documents
                    content_chunks = []
                    for i, result in enumerate(search_results):
                        content_text = None
                        
                        # Method 1: Check metadata content field (most common location)
                        if hasattr(result, "data") and isinstance(result.data, dict):
                            if "metadata" in result.data and isinstance(result.data["metadata"], dict):
                                metadata = result.data["metadata"]
                                
                                # Get the full content field
                                if "content" in metadata:
                                    content_text = str(metadata["content"]).strip()
                                    self.log(f"Extracted full content from document {i+1} metadata content ({len(content_text)} chars)")
                                
                                # Fallback: if no content field, try to get all metadata values
                                if not content_text:
                                    # Combine all metadata values
                                    metadata_parts = []
                                    for key, value in metadata.items():
                                        if value:
                                            metadata_parts.append(f"{key}: {value}")
                                    if metadata_parts:
                                        content_text = "\n".join(metadata_parts)
                                        self.log(f"Extracted content from document {i+1} metadata fields")
                        
                        # Method 2: Check page_content
                        if not content_text and hasattr(result, "page_content"):
                            content_text = str(result.page_content).strip()
                            self.log(f"Extracted content from document {i+1} page_content ({len(content_text)} chars)")
                        
                        # Method 3: Check if result has metadata attribute directly
                        if not content_text and hasattr(result, "metadata") and isinstance(result.metadata, dict):
                            if "content" in result.metadata:
                                content_text = str(result.metadata["content"]).strip()
                                self.log(f"Extracted content from document {i+1} result.metadata.content ({len(content_text)} chars)")
                        
                        # Method 4: Check data fields directly
                        if not content_text and hasattr(result, "data") and isinstance(result.data, dict):
                            # Try common content field names
                            for key in ["content", "text", "page_content", "data"]:
                                if key in result.data and result.data[key]:
                                    content_text = str(result.data[key]).strip()
                                    self.log(f"Extracted content from document {i+1} data[{key}] ({len(content_text)} chars)")
                                    break
                        
                        # Method 5: Check text attribute
                        if not content_text and hasattr(result, "text") and result.text:
                            content_text = str(result.text).strip()
                            self.log(f"Extracted content from document {i+1} text attribute ({len(content_text)} chars)")
                        
                        # If we found content, add it to the list (avoid duplicates)
                        if content_text and content_text.strip() and content_text not in content_chunks:
                            content_chunks.append(content_text.strip())
                            self.log(f"Added content chunk {len(content_chunks)}: {content_text[:200]}...")
                    
                    # Return all content chunks joined together
                    if content_chunks:
                        # Join all content chunks with a clear separator
                        separator = "\n\n" + "="*80 + "\n\n"
                        content_parts = [f"{'='*80}\n--- Content Chunk {i+1} ---\n{'='*80}\n{c}" for i, c in enumerate(content_chunks)]
                        all_content = separator.join(content_parts)
                        self.log(f"Returning {len(content_chunks)} content chunk(s) for category '{category}'")
                        self.status = f"Category: {category} ({len(content_chunks)} chunks)"
                        return Message(text=all_content)
                    else:
                        # Fallback: return category name if we can't extract content
                        self.log(f"Category '{category}' found but no content extracted. Result structure: {[type(r).__name__ for r in search_results]}")
                        self.status = f"Category: {category}"
                        return Message(text=category)
            
            # If no category found, return default
            self.log(f"No category found in search results. Searched for: {categories}")
            self.log(f"Combined text sample: {combined_text_lower[:200]}")
            self.status = "Category: Clarification needed"
            return Message(text="Clarification needed")
            
        except Exception as e:
            error_msg = f"Error during search: {e}"
            self.log(error_msg)
            self.status = error_msg
            # Log the full exception for debugging
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            return Message(text="Clarification needed")
