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
        MessageTextInput(
            name="collection_name",
            display_name="Collection Name",
            info="Name of the Chroma collection to search. Must match the collection name used during ingestion.",
            required=False,
            show=True,
            value="langflow",
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

        # Get collection_name - handle both string and Message inputs
        collection_name_value = self.collection_name
        if isinstance(collection_name_value, str):
            collection_name = collection_name_value
        elif hasattr(collection_name_value, "text"):
            # Handle Message objects
            collection_name = str(collection_name_value.text)
        elif hasattr(collection_name_value, "data") and isinstance(collection_name_value.data, dict):
            # Handle Data objects
            collection_name = str(collection_name_value.data.get("text", collection_name_value.data.get("content", str(collection_name_value))))
        else:
            collection_name = str(collection_name_value) if collection_name_value else "langflow"
        
        # Use default if empty
        if not collection_name or not collection_name.strip():
            collection_name = "langflow"

        # Build Chroma instance
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
            self.log(f"Connected to Chroma DB at '{persist_directory}' (collection: '{collection_name}', documents: {num_docs})")
        except Exception as e:
            self.log(f"Warning: Could not retrieve collection info: {e}")

        return chroma

    @override
    def search_documents(self) -> Message:
        """Search for documents in the Chroma vector store and return search results."""
        try:
            # Ensure vector store is built
            if self._cached_vector_store is not None:
                vector_store = self._cached_vector_store
            else:
                vector_store = self.build_vector_store()
                self._cached_vector_store = vector_store

            # Get search_query from parent class inputs
            search_query: str = self.search_query
            if not search_query:
                self.status = "No search query provided."
                return Message(text="")

            self.log(f"Searching Chroma DB with query: '{search_query}'")
            self.log(f"Number of results requested: {self.number_of_results}")

            # Use Chroma's native similarity_search_with_score method
            docs = []
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
                        return Message(text="")
                    
                    # Extract documents from (doc, score) tuples
                    docs = [doc for doc, score in docs_with_scores]
                    
                elif hasattr(vector_store, "similarity_search"):
                    # Fallback: try similarity_search if similarity_search_with_score doesn't exist
                    self.log("similarity_search_with_score not available, trying similarity_search")
                    docs = vector_store.similarity_search(
                        query=search_query,
                        k=self.number_of_results,
                    )
                    
                    self.log(f"Found {len(docs)} documents")
                else:
                    error_msg = "Vector store does not have similarity_search or similarity_search_with_score methods"
                    self.log(error_msg)
                    self.status = error_msg
                    return Message(text="")
                    
            except Exception as search_error:
                error_msg = f"Error during similarity search: {search_error}"
                self.log(error_msg)
                import traceback
                self.log(f"Search error traceback: {traceback.format_exc()}")
                self.status = error_msg
                return Message(text="")
            
            if not docs:
                self.log("No results found for the query.")
                self.status = "No results found."
                return Message(text="")
            
            # Convert documents to text content
            content_parts = []
            for i, doc in enumerate(docs):
                # Extract text content from document
                text_content = ""
                if hasattr(doc, "page_content"):
                    text_content = doc.page_content
                elif hasattr(doc, "text"):
                    text_content = doc.text
                elif isinstance(doc, str):
                    text_content = doc
                
                if text_content:
                    content_parts.append(f"--- Result {i+1} ---\n{text_content}")
            
            # Join all results
            if content_parts:
                result_text = "\n\n".join(content_parts)
                self.log(f"Returning {len(content_parts)} search result(s)")
                self.status = f"Found {len(content_parts)} result(s)"
                return Message(text=result_text)
            else:
                self.log("No content extracted from search results.")
                self.status = "No results found."
                return Message(text="")
            
        except Exception as e:
            error_msg = f"Error during search: {e}"
            self.log(error_msg)
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            self.status = error_msg
            return Message(text="")
