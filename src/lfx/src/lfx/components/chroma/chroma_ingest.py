from copy import deepcopy
from typing import TYPE_CHECKING

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing_extensions import override

from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.base.vectorstores.utils import chroma_collection_to_data
from lfx.helpers.data import docs_to_data
from lfx.inputs.inputs import BoolInput, DropdownInput, FloatInput, HandleInput, IntInput, StrInput
from lfx.schema.data import Data

if TYPE_CHECKING:
    from lfx.schema.dataframe import DataFrame


class ChromaVectorStoreComponent(LCVectorStoreComponent):
    """Chroma Vector Store with search capabilities."""

    display_name: str = "Chroma DB"
    description: str = "Chroma Vector Store with search capabilities"
    name = "Chroma"
    icon = "Chroma"

    inputs = [
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            value="langflow",
        ),
        StrInput(
            name="persist_directory",
            display_name="Persist Directory",
        ),
        *LCVectorStoreComponent.inputs,
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        StrInput(
            name="chroma_server_cors_allow_origins",
            display_name="Server CORS Allow Origins",
            advanced=True,
        ),
        StrInput(
            name="chroma_server_host",
            display_name="Server Host",
            advanced=True,
        ),
        IntInput(
            name="chroma_server_http_port",
            display_name="Server HTTP Port",
            advanced=True,
        ),
        IntInput(
            name="chroma_server_grpc_port",
            display_name="Server gRPC Port",
            advanced=True,
        ),
        BoolInput(
            name="chroma_server_ssl_enabled",
            display_name="Server SSL Enabled",
            advanced=True,
        ),
        BoolInput(
            name="allow_duplicates",
            display_name="Allow Duplicates",
            advanced=True,
            info="If false, will not add documents that are already in the Vector Store.",
        ),
        DropdownInput(
            name="search_method",
            display_name="Search Method",
            options=["Vector Search", "Hybrid Search"],
            value="Vector Search",
            advanced=True,
            info="Vector Search uses semantic similarity only. Hybrid Search combines vector similarity with keyword/BM25 search.",
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type",
            options=["Similarity", "MMR"],
            value="Similarity",
            advanced=True,
            info="Search type for vector search. Only used when Search Method is Vector Search.",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            info="Number of results to return.",
            advanced=True,
            value=10,
        ),
        FloatInput(
            name="vector_weight",
            display_name="Vector Weight",
            value=0.7,
            advanced=True,
            info="Weight for vector similarity search in hybrid mode (0.0-1.0). Higher values prioritize semantic similarity.",
        ),
        FloatInput(
            name="keyword_weight",
            display_name="Keyword Weight",
            value=0.3,
            advanced=True,
            info="Weight for keyword/BM25 search in hybrid mode (0.0-1.0). Higher values prioritize exact term matching.",
        ),
        IntInput(
            name="hybrid_search_limit",
            display_name="Hybrid Search Limit",
            value=100,
            advanced=True,
            info="Number of candidates to retrieve from each search method before fusion in hybrid search.",
        ),
        IntInput(
            name="limit",
            display_name="Limit",
            advanced=True,
            info="Limit the number of records to compare when Allow Duplicates is False.",
        ),
    ]

    @override
    @check_cached_vector_store
    def build_vector_store(self) -> Chroma:
        """Builds the Chroma object."""
        try:
            from chromadb import Client, PersistentClient
            from langchain_chroma import Chroma
        except ImportError as e:
            msg = "Could not import Chroma integration package. Please install it with `pip install langchain-chroma`."
            raise ImportError(msg) from e
        # Chroma settings
        chroma_settings = None
        client = None
        if self.chroma_server_host:
            chroma_settings = Settings(
                chroma_server_cors_allow_origins=self.chroma_server_cors_allow_origins or [],
                chroma_server_host=self.chroma_server_host,
                chroma_server_http_port=self.chroma_server_http_port or None,
                chroma_server_grpc_port=self.chroma_server_grpc_port or None,
                chroma_server_ssl_enabled=self.chroma_server_ssl_enabled,
            )
            client = Client(settings=chroma_settings)

        # Check persist_directory and expand it if it is a relative path
        persist_directory = self.resolve_path(self.persist_directory) if self.persist_directory is not None else None

        # Create Chroma instance first (may load existing collection)
        chroma = Chroma(
            persist_directory=persist_directory,
            client=client,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

        # Only clear collection when ingesting new data (not when just searching/reloading)
        # Check if we have data to ingest - be very explicit about this check
        ingest_data: list | Data | DataFrame = self.ingest_data
        
        # Check if we're in search-only mode (has search query but no ingest data)
        has_search_query = bool(getattr(self, 'search_query', None) and str(self.search_query).strip())
        
        # More robust check: only consider it ingestion if we have actual non-empty data
        # AND we're not in search-only mode
        has_data_to_ingest = False
        if ingest_data is not None and not has_search_query:
            if isinstance(ingest_data, list):
                # Empty list means no data to ingest
                has_data_to_ingest = len(ingest_data) > 0
            elif hasattr(ingest_data, 'empty'):
                # DataFrame case - check if empty
                has_data_to_ingest = not ingest_data.empty
            else:
                # Data object or other - check if it's truthy and has content
                has_data_to_ingest = bool(ingest_data) and (
                    (hasattr(ingest_data, 'data') and ingest_data.data) or
                    (hasattr(ingest_data, 'get_text') and ingest_data.get_text())
                )
        
        self.log(f"Checking ingestion mode: ingest_data={type(ingest_data).__name__}, has_search_query={has_search_query}, has_data_to_ingest={has_data_to_ingest}")

        if has_data_to_ingest:
            # Clear ALL existing documents from the collection to ensure fresh start
            # This prevents any duplicates from previous ingestions
            self.log(f"Detected new data to ingest. Clearing existing collection '{self.collection_name}' to ensure fresh start.")
            try:
                # Get all existing documents BEFORE clearing
                existing_docs = chroma.get()
                if existing_docs and existing_docs.get("ids"):
                    existing_ids = existing_docs["ids"]
                    existing_documents = existing_docs.get("documents", [])
                    existing_metadatas = existing_docs.get("metadatas", [])
                    
                    if existing_ids:
                        # Log what we're about to clear
                        self.log(f"BEFORE CLEARING: Found {len(existing_ids)} existing documents in collection '{self.collection_name}'")
                        # Show sample of what's being cleared (first 3 documents)
                        for i, doc_id in enumerate(existing_ids[:3]):
                            doc_text = existing_documents[i] if i < len(existing_documents) else "N/A"
                            metadata = existing_metadatas[i] if i < len(existing_metadatas) else {}
                            # Truncate long text for logging
                            text_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                            self.log(f"  Document {i+1} (ID: {doc_id[:20]}...): text_preview='{text_preview}', metadata_keys={list(metadata.keys())}")
                        
                        if len(existing_ids) > 3:
                            self.log(f"  ... and {len(existing_ids) - 3} more documents")
                        
                        # Delete all existing documents by their IDs
                        chroma.delete(ids=existing_ids)
                        self.log(f"AFTER CLEARING: Deleted {len(existing_ids)} documents from collection '{self.collection_name}'")
                        
                        # Verify collection is empty
                        verify_docs = chroma.get()
                        verify_count = len(verify_docs.get("ids", [])) if verify_docs else 0
                        self.log(f"VERIFICATION: Collection now contains {verify_count} documents (should be 0)")
                    else:
                        self.log(f"Collection '{self.collection_name}' is empty, ready for fresh data.")
                else:
                    self.log(f"Collection '{self.collection_name}' is empty, ready for fresh data.")
            except Exception as e:
                # If clearing fails, try alternative method: delete collection and recreate
                self.log(f"Warning: Could not clear documents via delete: {e}. Attempting to delete collection...")
                try:
                    # Try to delete the entire collection via the underlying client
                    if hasattr(chroma, "_client") and chroma._client is not None:
                        chroma._client.delete_collection(name=self.collection_name)
                        self.log(f"Deleted collection '{self.collection_name}' via client.")
                        # Recreate Chroma instance with fresh collection
                        chroma = Chroma(
                            persist_directory=persist_directory,
                            client=client,
                            embedding_function=self.embedding,
                            collection_name=self.collection_name,
                        )
                    elif client is None and persist_directory:
                        # For local persist directory, create a client to delete collection
                        temp_client = PersistentClient(path=persist_directory)
                        try:
                            temp_client.delete_collection(name=self.collection_name)
                            self.log(f"Deleted collection '{self.collection_name}' via PersistentClient.")
                            # Recreate Chroma instance with fresh collection
                            chroma = Chroma(
                                persist_directory=persist_directory,
                                client=client,
                                embedding_function=self.embedding,
                                collection_name=self.collection_name,
                            )
                        except Exception as del_e:
                            self.log(f"Could not delete collection: {del_e}")
                    elif client is not None:
                        # For server-based client, delete collection directly
                        try:
                            client.delete_collection(name=self.collection_name)
                            self.log(f"Deleted collection '{self.collection_name}' via server client.")
                            # Recreate Chroma instance with fresh collection
                            chroma = Chroma(
                                persist_directory=persist_directory,
                                client=client,
                                embedding_function=self.embedding,
                                collection_name=self.collection_name,
                            )
                        except Exception as del_e:
                            self.log(f"Could not delete collection: {del_e}")
                except Exception as del_e2:
                    self.log(f"Warning: All deletion methods failed: {del_e2}. Will rely on duplicate checking.")
        else:
            # No data to ingest, just use existing collection for searching
            self.log(f"No new data to ingest. Using existing collection '{self.collection_name}' for search operations.")
            # Log current collection state for reference
            try:
                current_docs = chroma.get()
                current_count = len(current_docs.get("ids", [])) if current_docs else 0
                self.log(f"CURRENT STATE: Collection '{self.collection_name}' contains {current_count} document(s) (not modified)")
                if current_count > 0:
                    # Show sample of existing documents
                    current_ids = current_docs.get("ids", [])[:3]
                    current_documents = current_docs.get("documents", [])
                    current_metadatas = current_docs.get("metadatas", [])
                    self.log(f"Sample of existing documents (first {min(3, len(current_ids))}):")
                    for i, doc_id in enumerate(current_ids):
                        doc_text = current_documents[i] if i < len(current_documents) else "N/A"
                        metadata = current_metadatas[i] if i < len(current_metadatas) else {}
                        text_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        self.log(f"  Document {i+1} (ID: {doc_id[:20]}...): text_preview='{text_preview}', metadata_keys={list(metadata.keys())}")
            except Exception as e:
                self.log(f"Could not retrieve collection state: {e}")

        self._add_documents_to_vector_store(chroma)
        
        # Log final collection state
        limit = int(self.limit) if self.limit is not None and str(self.limit).strip() else None
        final_collection = chroma.get(limit=limit)
        final_count = len(final_collection.get("ids", [])) if final_collection else 0
        self.log(f"BUILD COMPLETE: Collection '{self.collection_name}' final state: {final_count} document(s)")
        
        self.status = chroma_collection_to_data(final_collection)
        return chroma

    def _add_documents_to_vector_store(self, vector_store: "Chroma") -> None:
        """Adds documents to the Vector Store."""
        ingest_data: list | Data | DataFrame = self.ingest_data
        if not ingest_data:
            self.status = ""
            self.log("No ingest_data provided, skipping document addition.")
            return

        # Convert DataFrame to Data if needed using parent's method
        ingest_data = self._prepare_ingest_data()
        
        self.log(f"PREPARING TO ADD: Processing {len(ingest_data) if isinstance(ingest_data, list) else 1} data item(s) for ingestion")

        stored_documents_without_id = []
        if self.allow_duplicates:
            stored_data = []
            self.log("Duplicate checking disabled (allow_duplicates=True)")
        else:
            limit = int(self.limit) if self.limit is not None and str(self.limit).strip() else None
            stored_data = chroma_collection_to_data(vector_store.get(limit=limit))
            self.log(f"Checking for duplicates: Found {len(stored_data)} existing documents in collection")
            for value in deepcopy(stored_data):
                del value.id
                stored_documents_without_id.append(value)

        documents = []
        skipped_duplicates = 0
        for i, _input in enumerate(ingest_data or []):
            if isinstance(_input, Data):
                if _input not in stored_documents_without_id:
                    # Handle incoming row data: if 'content' field exists, use it as page_content
                    page_content = None
                    metadata = {}
                    
                    if 'content' in _input.data and isinstance(_input.data['content'], str):
                        # Use the content field as page_content
                        page_content = _input.data['content']
                        # Copy all other fields as metadata (except content)
                        metadata = {k: v for k, v in _input.data.items() if k != 'content'}
                    else:
                        # Use standard conversion via to_lc_document()
                        doc = _input.to_lc_document()
                        page_content = doc.page_content
                        metadata = doc.metadata.copy() if doc.metadata else {}
                    
                    # Create Document with the determined content
                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)
                    
                    # Log first 3 documents being added
                    if len(documents) <= 3:
                        text_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        metadata_keys = list(doc.metadata.keys()) if doc.metadata else []
                        self.log(f"  Document {len(documents)} to add: text_preview='{text_preview}', metadata_keys={metadata_keys}")
                else:
                    skipped_duplicates += 1
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if skipped_duplicates > 0:
            self.log(f"Skipped {skipped_duplicates} duplicate document(s) (already exist in collection)")

        if documents and self.embedding is not None:
            self.log(f"ADDING DOCUMENTS: About to add {len(documents)} new document(s) to the Vector Store")
            # Filter complex metadata to prevent ChromaDB errors
            try:
                from langchain_community.vectorstores.utils import filter_complex_metadata

                filtered_documents = filter_complex_metadata(documents)
                if len(filtered_documents) != len(documents):
                    self.log(f"Filtered {len(documents) - len(filtered_documents)} document(s) with complex metadata")
                vector_store.add_documents(filtered_documents)
                self.log(f"SUCCESS: Added {len(filtered_documents)} document(s) to collection '{self.collection_name}'")
                
                # Verify what's in the collection now
                final_docs = vector_store.get()
                final_count = len(final_docs.get("ids", [])) if final_docs else 0
                self.log(f"FINAL STATE: Collection '{self.collection_name}' now contains {final_count} total document(s)")
                
                # Show sample of final documents (last 3 added)
                if final_docs and final_docs.get("ids"):
                    final_ids = final_docs.get("ids", [])
                    final_documents = final_docs.get("documents", [])
                    final_metadatas = final_docs.get("metadatas", [])
                    # Show last 3 documents
                    start_idx = max(0, len(final_ids) - 3)
                    self.log(f"Sample of final documents (last {min(3, len(final_ids))}):")
                    for i in range(start_idx, len(final_ids)):
                        idx = i - start_idx
                        doc_id = final_ids[i]
                        doc_text = final_documents[i] if i < len(final_documents) else "N/A"
                        metadata = final_metadatas[i] if i < len(final_metadatas) else {}
                        text_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        self.log(f"  Document {idx+1} (ID: {doc_id[:20]}...): text_preview='{text_preview}', metadata_keys={list(metadata.keys())}")
            except ImportError:
                self.log("Warning: Could not import filter_complex_metadata. Adding documents without filtering.")
                vector_store.add_documents(documents)
                self.log(f"SUCCESS: Added {len(documents)} document(s) to collection '{self.collection_name}'")
                
                # Verify what's in the collection now
                final_docs = vector_store.get()
                final_count = len(final_docs.get("ids", [])) if final_docs else 0
                self.log(f"FINAL STATE: Collection '{self.collection_name}' now contains {final_count} total document(s)")
        else:
            if not documents:
                self.log("No documents to add to the Vector Store (all were duplicates or empty).")
            elif not self.embedding:
                self.log("No embedding function provided, cannot add documents to the Vector Store.")

    @override
    def search_documents(self) -> list[Data]:
        """Search for documents in the vector store with optional hybrid search."""
        if self._cached_vector_store is not None:
            vector_store = self._cached_vector_store
        else:
            vector_store = self.build_vector_store()
            self._cached_vector_store = vector_store

        search_query: str = self.search_query
        if not search_query:
            self.status = ""
            return []

        self.log(f"Search input: {search_query}")
        self.log(f"Search method: {self.search_method}")
        self.log(f"Number of results: {self.number_of_results}")

        # Use hybrid search if enabled
        if getattr(self, "search_method", "Vector Search") == "Hybrid Search":
            return self._hybrid_search(vector_store, search_query)

        # Use standard vector search
        self.log(f"Search type: {self.search_type}")
        search_results = self.search_with_vector_store(
            search_query, self.search_type, vector_store, k=self.number_of_results
        )
        self.status = search_results
        return search_results

    def _hybrid_search(self, vector_store: Chroma, query: str) -> list[Data]:
        """Perform hybrid search combining vector similarity and keyword search."""
        try:
            # Try ChromaDB native hybrid search first
            return self._chroma_native_hybrid_search(vector_store, query)
        except (AttributeError, ImportError, TypeError) as e:
            # These exceptions indicate the native API is not available or incompatible
            self.log(f"ChromaDB native hybrid search not available: {e}. Falling back to EnsembleRetriever.")
            # Fall back to EnsembleRetriever pattern
            return self._ensemble_retriever_hybrid_search(vector_store, query)
        except ValueError as e:
            # ValueError for missing embedding should propagate (configuration error)
            # But other ValueErrors (e.g., invalid API usage) can fall back
            if "embedding" in str(e).lower() or "required" in str(e).lower():
                raise  # Re-raise configuration errors
            self.log(f"Error during ChromaDB native hybrid search: {e}. Falling back to EnsembleRetriever.")
            return self._ensemble_retriever_hybrid_search(vector_store, query)
        except Exception as e:
            # Catch any other runtime errors during native hybrid search execution
            # (e.g., errors calling hybrid_search, result parsing issues, etc.)
            self.log(f"Error during ChromaDB native hybrid search: {e}. Falling back to EnsembleRetriever.")
            return self._ensemble_retriever_hybrid_search(vector_store, query)

    def _chroma_native_hybrid_search(self, vector_store: Chroma, query: str) -> list[Data]:
        """Use ChromaDB's native hybrid search API if available."""
        try:
            from chromadb import K, Knn, Rrf, Search
        except ImportError as e:
            msg = "ChromaDB native hybrid search requires chromadb package with hybrid search support."
            raise ImportError(msg) from e

        # Access underlying Chroma collection
        collection = vector_store._collection  # noqa: SLF001

        # Check if hybrid_search method exists
        if not hasattr(collection, "hybrid_search"):
            raise AttributeError("ChromaDB collection does not support hybrid_search method.")

        # Get weights (normalize if needed)
        vector_weight = float(getattr(self, "vector_weight", 0.7))
        keyword_weight = float(getattr(self, "keyword_weight", 0.3))
        hybrid_limit = int(getattr(self, "hybrid_search_limit", 100))
        num_results = int(self.number_of_results)

        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight

        # Embed the query for vector search
        if not self.embedding:
            raise ValueError("Embedding is required for hybrid search.")

        # ChromaDB's Knn accepts query (text string) not query_embeddings
        # The embedding function is handled by the collection
        # Build hybrid rank using RRF (Reciprocal Rank Fusion)
        # Note: ChromaDB API - Knn accepts query (text) not embeddings
        try:
            # Try with sparse embedding key (requires ChromaDB Cloud with sparse support)
            # For open-source ChromaDB, this will fail and fall back to single dense query
            hybrid_rank = Rrf(
                ranks=[
                    Knn(query=query, return_rank=True, limit=hybrid_limit),
                    Knn(query=query, key="sparse_embedding", return_rank=True, limit=hybrid_limit),
                ],
                weights=[vector_weight, keyword_weight],
                k=num_results,
            )
        except (TypeError, AttributeError, ValueError) as e:
            # Fallback: use only dense vector search if sparse not available
            # This happens with open-source ChromaDB which doesn't support sparse embeddings
            self.log(f"Sparse embedding not available ({e}), using vector search only.")
            hybrid_rank = Rrf(
                ranks=[
                    Knn(query=query, return_rank=True, limit=hybrid_limit),
                ],
                weights=[1.0],
                k=num_results,
            )

        # Build search query
        search = (
            Search()
            .rank(hybrid_rank)
            .limit(num_results)
            .select(K.DOCUMENT, K.SCORE)
        )

        # Execute hybrid search
        try:
            results = collection.hybrid_search(search)
        except Exception as e:
            # If hybrid_search fails, it might be due to API incompatibility
            raise RuntimeError(f"ChromaDB hybrid_search failed: {e}") from e

        # Convert results to Data objects
        # ChromaDB returns results in format: {"documents": [[...]], "ids": [[...]], "metadatas": [[...]], "distances": [[...]]}
        data_list = []
        
        # Handle nested list structure (ChromaDB returns lists of lists)
        documents = results.get("documents", [])
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        scores = results.get("scores", distances)  # Use distances as scores if scores not available

        # Flatten nested structure if needed
        if documents and isinstance(documents[0], list):
            documents = documents[0]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if metadatas and isinstance(metadatas[0], list):
            metadatas = metadatas[0]
        if scores and isinstance(scores[0], list):
            scores = scores[0]

        # Convert distances to similarity scores (ChromaDB returns distances, lower is better)
        # For cosine similarity, convert distance to score: score = 1 - distance
        if scores and len(scores) > 0:
            # Check if these are distances (values typically 0-2 for cosine) or scores
            max_val = max(scores) if scores else 1.0
            if max_val > 1.0:
                # Likely distances, convert to scores
                scores = [1.0 - d if d <= 1.0 else 0.0 for d in scores]

        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids and i < len(ids) else None
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            score = scores[i] if scores and i < len(scores) else None

            data_dict = {
                "text": doc,
            }
            if metadata:
                data_dict.update(metadata)
            if doc_id:
                data_dict["id"] = doc_id
            if score is not None:
                data_dict["score"] = float(score)

            data_list.append(Data(**data_dict))

        self.status = data_list
        return data_list

    def _ensemble_retriever_hybrid_search(self, vector_store: Chroma, query: str) -> list[Data]:
        """Fallback hybrid search using EnsembleRetriever with BM25Retriever."""
        try:
            from langchain.retrievers import BM25Retriever, EnsembleRetriever
            from langchain_core.documents import Document
        except ImportError as e:
            msg = "EnsembleRetriever hybrid search requires langchain package. Please install it with `pip install langchain`."
            raise ImportError(msg) from e

        # Get weights
        vector_weight = float(getattr(self, "vector_weight", 0.7))
        keyword_weight = float(getattr(self, "keyword_weight", 0.3))
        num_results = int(self.number_of_results)

        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight

        # Get documents from Chroma for BM25 indexing
        try:
            chroma_results = vector_store.get(include=["documents", "metadatas"])
            documents = []
            for i, doc_text in enumerate(chroma_results.get("documents", [])):
                metadata = chroma_results.get("metadatas", [{}])[i] if chroma_results.get("metadatas") else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))
        except Exception as e:
            self.log(f"Error retrieving documents for BM25: {e}. Using vector search only.")
            return self.search_with_vector_store(query, self.search_type, vector_store, k=num_results)

        if not documents:
            self.log("No documents found in vector store.")
            return []

        # Build BM25 retriever
        try:
            bm25_retriever = BM25Retriever.from_documents(documents, k=num_results)
        except Exception as e:
            self.log(f"Error building BM25 retriever: {e}. Using vector search only.")
            return self.search_with_vector_store(query, self.search_type, vector_store, k=num_results)

        # Build vector retriever
        vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": num_results},
        )

        # Combine retrievers
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[vector_weight, keyword_weight],
        )

        # Perform hybrid search
        try:
            docs = ensemble.invoke(query)
            data_list = docs_to_data(docs)
            self.status = data_list
            return data_list
        except Exception as e:
            self.log(f"Error in ensemble retriever search: {e}. Falling back to vector search.")
            return self.search_with_vector_store(query, self.search_type, vector_store, k=num_results)
