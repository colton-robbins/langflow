from copy import deepcopy
from typing import TYPE_CHECKING

from chromadb.config import Settings
from langchain_chroma import Chroma
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

    def _log_vector_store_contents(self, vector_store: Chroma, context: str = "") -> None:
        """Log detailed information about the current state of the vector store."""
        try:
            # Get all documents from the collection
            limit = int(self.limit) if self.limit is not None and str(self.limit).strip() else None
            collection_data = vector_store.get(limit=limit)
            
            # Extract counts and data
            documents = collection_data.get("documents", [])
            ids = collection_data.get("ids", [])
            metadatas = collection_data.get("metadatas", [])
            
            total_count = len(documents) if documents else 0
            
            # Log header
            context_str = f" ({context})" if context else ""
            self.log(f"=== Vector Store Contents{context_str} ===")
            self.log(f"Collection Name: {self.collection_name}")
            self.log(f"Total Documents: {total_count}")
            
            if total_count == 0:
                self.log("Vector store is empty.")
                self.log("=" * 50)
                return
            
            # Log document IDs (sample if too many)
            if ids:
                max_ids_to_show = 20
                if len(ids) <= max_ids_to_show:
                    self.log(f"Document IDs ({len(ids)}): {ids}")
                else:
                    self.log(f"Document IDs (showing first {max_ids_to_show} of {len(ids)}): {ids[:max_ids_to_show]}")
                    self.log(f"... and {len(ids) - max_ids_to_show} more IDs")
            
            # Analyze metadata
            if metadatas:
                # Collect all unique metadata keys
                all_metadata_keys = set()
                for metadata in metadatas:
                    if metadata:
                        all_metadata_keys.update(metadata.keys())
                
                if all_metadata_keys:
                    self.log(f"Metadata Keys Found: {sorted(all_metadata_keys)}")
                    
                    # Show sample metadata values for each key
                    for key in sorted(all_metadata_keys):
                        values = []
                        for metadata in metadatas:
                            if metadata and key in metadata:
                                value = metadata[key]
                                if value not in values:
                                    values.append(value)
                                if len(values) >= 5:  # Show up to 5 unique values per key
                                    break
                        if values:
                            values_str = ", ".join([str(v) for v in values[:5]])
                            if len(values) > 5:
                                values_str += f" ... ({len(values)} total unique values)"
                            self.log(f"  - {key}: {values_str}")
            
            # Show sample document text (first line only)
            if documents:
                max_docs_to_show = 5
                self.log(f"Sample Documents (showing first {min(max_docs_to_show, len(documents))}):")
                for i, doc in enumerate(documents[:max_docs_to_show]):
                    doc_str = str(doc)
                    # Get first line only
                    first_line = doc_str.split('\n')[0] if '\n' in doc_str else doc_str
                    # Truncate if still too long
                    doc_preview = first_line[:100] + "..." if len(first_line) > 100 else first_line
                    doc_id = ids[i] if ids and i < len(ids) else "N/A"
                    self.log(f"  [{doc_id}] {doc_preview}")
                if len(documents) > max_docs_to_show:
                    self.log(f"  ... and {len(documents) - max_docs_to_show} more documents")
            
            # Log collection statistics if available
            try:
                collection = vector_store._collection  # noqa: SLF001
                if hasattr(collection, "count"):
                    count = collection.count()
                    self.log(f"Collection Count (via API): {count}")
            except Exception:
                pass  # Count API might not be available
            
            self.log("=" * 50)
        except Exception as e:
            self.log(f"Error logging vector store contents: {e}")

    @override
    @check_cached_vector_store
    def build_vector_store(self) -> Chroma:
        """Builds the Chroma object."""
        try:
            from chromadb import Client
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

        chroma = Chroma(
            persist_directory=persist_directory,
            client=client,
            embedding_function=self.embedding,
            collection_name=self.collection_name,
        )

        # Log vector store contents before adding new documents
        self._log_vector_store_contents(chroma, "before adding new documents")

        self._add_documents_to_vector_store(chroma)
        
        # Log vector store contents after adding documents
        self._log_vector_store_contents(chroma, "after adding documents")
        
        limit = int(self.limit) if self.limit is not None and str(self.limit).strip() else None
        self.status = chroma_collection_to_data(chroma.get(limit=limit))
        return chroma

    def _add_documents_to_vector_store(self, vector_store: "Chroma") -> None:
        """Adds documents to the Vector Store."""
        ingest_data: list | Data | DataFrame = self.ingest_data
        if not ingest_data:
            self.status = ""
            self.log("No ingest data provided. Vector store contents unchanged.")
            return

        # Convert DataFrame to Data if needed using parent's method
        ingest_data = self._prepare_ingest_data()

        stored_documents_without_id = []
        if self.allow_duplicates:
            stored_data = []
            self.log("Allow Duplicates is enabled. All provided documents will be added.")
        else:
            limit = int(self.limit) if self.limit is not None and str(self.limit).strip() else None
            stored_data = chroma_collection_to_data(vector_store.get(limit=limit))
            self.log(f"Checking for duplicates against {len(stored_data)} existing documents in vector store.")
            for value in deepcopy(stored_data):
                del value.id
                stored_documents_without_id.append(value)

        documents = []
        skipped_count = 0
        for _input in ingest_data or []:
            if isinstance(_input, Data):
                if _input not in stored_documents_without_id:
                    documents.append(_input.to_lc_document())
                else:
                    skipped_count += 1
            else:
                msg = "Vector Store Inputs must be Data objects."
                raise TypeError(msg)

        if skipped_count > 0:
            self.log(f"Skipped {skipped_count} duplicate document(s) (already exist in vector store).")

        if documents and self.embedding is not None:
            self.log(f"Adding {len(documents)} new document(s) to the Vector Store.")
            # Filter complex metadata to prevent ChromaDB errors
            try:
                from langchain_community.vectorstores.utils import filter_complex_metadata

                filtered_documents = filter_complex_metadata(documents)
                vector_store.add_documents(filtered_documents)
                self.log(f"Successfully added {len(filtered_documents)} document(s) to vector store.")
            except ImportError:
                self.log("Warning: Could not import filter_complex_metadata. Adding documents without filtering.")
                vector_store.add_documents(documents)
                self.log(f"Successfully added {len(documents)} document(s) to vector store.")
        else:
            if not documents:
                self.log("No new documents to add to the Vector Store (all were duplicates or empty).")
            elif self.embedding is None:
                self.log("Cannot add documents: embedding function is not set.")

    @override
    def search_documents(self) -> list[Data]:
        """Search for documents in the vector store with optional hybrid search."""
        if self._cached_vector_store is not None:
            vector_store = self._cached_vector_store
        else:
            vector_store = self.build_vector_store()
            self._cached_vector_store = vector_store

        # Log vector store contents before searching
        self._log_vector_store_contents(vector_store, "before search")

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
        self.log(f"Search completed. Found {len(search_results)} result(s).")
        if search_results:
            self.log("Search result IDs: " + ", ".join([str(r.id) if hasattr(r, 'id') and r.id else f"result_{i}" for i, r in enumerate(search_results[:10])]))
            if len(search_results) > 10:
                self.log(f"... and {len(search_results) - 10} more results")
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

        self.log(f"Hybrid search (native) completed. Found {len(data_list)} result(s).")
        if data_list:
            result_ids = [str(d.id) if hasattr(d, 'id') and d.id else f"result_{i}" for i, d in enumerate(data_list[:10])]
            self.log("Search result IDs: " + ", ".join(result_ids))
            if len(data_list) > 10:
                self.log(f"... and {len(data_list) - 10} more results")
            # Log score range if available
            scores = [d.score for d in data_list if hasattr(d, 'score') and d.score is not None]
            if scores:
                self.log(f"Score range: {min(scores):.4f} - {max(scores):.4f}")

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
            self.log(f"Performing ensemble hybrid search with {len(documents)} documents indexed for BM25.")
            docs = ensemble.invoke(query)
            data_list = docs_to_data(docs)
            self.log(f"Hybrid search (ensemble) completed. Found {len(data_list)} result(s).")
            if data_list:
                result_ids = [str(d.id) if hasattr(d, 'id') and d.id else f"result_{i}" for i, d in enumerate(data_list[:10])]
                self.log("Search result IDs: " + ", ".join(result_ids))
                if len(data_list) > 10:
                    self.log(f"... and {len(data_list) - 10} more results")
            self.status = data_list
            return data_list
        except Exception as e:
            self.log(f"Error in ensemble retriever search: {e}. Falling back to vector search.")
            return self.search_with_vector_store(query, self.search_type, vector_store, k=num_results)
