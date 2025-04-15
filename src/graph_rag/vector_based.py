import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class VectorRAG:
    """Minimal RAG system for Building Information Modeling educational lab using OpenAI."""

    def __init__(
        self,
        embedding_files: List[Path],
        embeddings_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
    ):
        """Initialize RAG system with a list of embedding files using OpenAI.

        Args:
            embedding_files: List of paths to JSON files containing pre-computed OpenAI embeddings.
            embeddings_model: Name of the OpenAI model for query embeddings.
            llm_model: Name of the OpenAI chat model for the LLM.
        """
        # 1. Set up OpenAI embedding model for queries
        # Assumes OPENAI_API_KEY is set in environment
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)

        # 2. Load documents and create vector store using pre-computed embeddings if available
        self.vector_store = self._load_documents(embedding_files)

        # 3. Set up OpenAI LLM
        # Assumes OPENAI_API_KEY is set in environment
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)

        # 4. Create BIM-specific prompt template
        self.prompt = PromptTemplate(
            template="""Use the following context to answer the question about Building Information Modeling (BIM).
        Focus only on the information present in the text provided. If the context does not contain the answer, state that clearly.

        Context:
        {context}

        Question: {question}

        Answer:""",
            input_variables=["context", "question"],
        )

    def _load_documents(self, embedding_files: List[Path]):
        """Load documents and pre-computed embeddings from a list of JSON files into FAISS vector store.

        If embeddings exist in the JSON files, they'll be used directly instead of recomputing.

        Args:
            embedding_files: List of paths to JSON embedding files to load.
        """
        documents = []
        metadatas = []
        precomputed_embeddings: Optional[List[List[float]]] = []
        using_precomputed = False

        # Iterate over the provided list of files
        for file_path in embedding_files:
            if not file_path.is_file():
                print(f"Warning: Skipping non-existent file {file_path}")
                continue

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                    # Process chunks from each file
                    for chunk in data.get("chunks", []):
                        # Extract text and metadata
                        text = chunk.get("text")
                        if not text:
                            continue

                        metadata = {
                            "entity_name": chunk.get("entity_name", "Unknown Entity"),
                            "source_file": file_path.stem.replace("_embeddings", ".ttl"),
                        }

                        # Check if this chunk has a pre-computed embedding
                        if "embedding" in chunk and isinstance(chunk["embedding"], list):
                            precomputed_embeddings.append(chunk["embedding"])
                            using_precomputed = True
                        else:
                            # If any file lacks embeddings, we must recompute all for consistency
                            using_precomputed = False

                        documents.append(text)
                        metadatas.append(metadata)

            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file {file_path}")
                continue
            except Exception as e:
                print(f"Warning: Error processing file {file_path}: {e}")
                continue

        # Check if we actually processed any documents
        if not documents:
            print("No valid documents found in the provided files.")
            return None

        # Create vector store
        if using_precomputed and len(precomputed_embeddings) == len(documents):
            print(f"Using {len(precomputed_embeddings)} pre-computed embeddings from specified files")
            # Convert to numpy array
            embeddings_array = np.array(precomputed_embeddings, dtype=np.float32)
            # Create FAISS index from pre-computed embeddings
            vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(documents, embeddings_array)), embedding=self.embeddings, metadatas=metadatas
            )
            return vector_store
        else:
            print(
                f"Computing embeddings for {len(documents)} documents (pre-computed embeddings not found, incomplete, or inconsistent across files)"
            )
            # Ensure embeddings are computed only if documents exist
            if documents:
                return FAISS.from_texts(documents, self.embeddings, metadatas=metadatas)
            else:
                return None

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using RAG, retrieving top_k documents.

        Args:
            question: Question about BIM data
            top_k: The number of documents to retrieve for context.

        Returns:
            Dictionary with response and sources
        """
        # Check if vector store exists
        if not self.vector_store:
            return {"answer": "Vector store not initialized or no documents loaded.", "sources": []}

        # 1. Retrieve relevant documents dynamically
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"Error during document retrieval: {e}")
            return {"answer": "An error occurred during document retrieval.", "sources": []}

        if not docs:
            return {"answer": "No relevant documents found for the question.", "sources": []}

        # 2. Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. Generate answer using the LLM and prompt
        try:
            # Create the prompt input dictionary
            prompt_input = {"context": context, "question": question}
            # Format the prompt
            formatted_prompt = self.prompt.format(**prompt_input)
            # Invoke the LLM directly
            result = self.llm.invoke(formatted_prompt)
            answer = result.content  # Extract text content from AIMessage
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return {"answer": "An error occurred during answer generation.", "sources": []}

        # 4. Extract sources for transparency
        sources = []
        for doc in docs:
            sources.append(
                {
                    "entity": doc.metadata.get("entity_name", "Unknown"),
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "text": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                }
            )

        # Return formatted result
        return {"answer": answer, "sources": sources}
