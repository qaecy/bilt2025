# Graph RAG Querying Notebook - Presenter Notes

## Overview
This notebook demonstrates and compares two approaches to Retrieval Augmented Generation (RAG) for Building Information Modeling (BIM) data:
1.  **Vector-Based RAG**: Uses semantic search over pre-computed text embeddings of building entities.
2.  **Query-Based RAG**: Translates natural language questions into SPARQL queries executed against an RDF knowledge graph.

This is Part 2 of the lab, building upon the data processed and embedded in Part 1 (`01_data_integration.ipynb`).

## Presentation Flow

### Introduction (Title Slide)
- Welcome attendees back for Part 2.
- Briefly recap Part 1: IFC/PDF -> RDF Graph -> Embeddings.
- Introduce the goal for Part 2: Querying the processed data using two different RAG methods.
- Explain the two approaches (Vector vs. Graph/Query-based).
- **Presenter action:** Show the title slide and overview bullet points.

### Section 0: Setup
- Explain this cell is identical to Part 1's setup, ensuring the environment (Colab/local) is configured and dependencies are installed.
- **Presenter action:** Run the cell and verify successful setup.
- **Key talking point:** "We start with the same setup to ensure consistency and install necessary libraries like LangChain and pyoxigraph."

### Section 1: Import Libraries and Setup Paths
- Explain the necessary imports: `pandas` for display, `matplotlib` for optional plots, `Path` for paths, and our custom RAG classes (`VectorRAG` from `vector_based.py` and `QueryRAG` from `query_based.py`).
- Point out the path definitions for embeddings (`EMBEDDINGS_DIR`) and the graph data (`GRAPH_DIR`).
- **Presenter action:** Run the cell, confirm imports, and show the resolved paths.
- **Key talking point:** "We import our RAG implementations and set up paths to the data prepared in the previous lab."

### Section 2: Vector-Based RAG (Simple RAG with LangChain)
- **Goal:** Demonstrate RAG using semantic search over embeddings.
- **Subsection 2.1: Explore Available Embeddings**
    - Show the list of embedding files created in Part 1.
    - **Presenter action:** Run the cell, display the table of embedding files.
    - **Key talking point:** "These JSON files contain the text representations and vector embeddings for entities from our building models."
- **Subsection 2.2: Initialize RAG System**
    - Explain the `VectorRAG` class from `src.graph_rag.vector_based.py`.
    - It loads the embeddings into a FAISS vector store.
    - It uses an OpenAI model (`gpt-3.5-turbo` by default) and `OpenAIEmbeddings` for querying.
    - Mention the internal prompt template used.
    - **Presenter action:** Run the cell to initialize the `VectorRAG` instance (now named `vector_rag`). Note the messages about using pre-computed embeddings.
    - **Key talking point:** "We initialize our first RAG system, loading the embeddings for fast semantic search."
- **Subsection 2.3: Ask Questions**
    - Show the example questions.
    - Explain the `vector_rag.query(question, top_k=k)` method.
    - It retrieves the top `k` relevant text chunks (context) based on semantic similarity to the question's embedding.
    - The context and question are sent to the LLM via a prompt template to generate an answer.
    - The sources (retrieved chunks) are returned for transparency.
    - **Presenter action:** Run the cell. Discuss the answers and the source tables provided for each question.
    - **Key talking point:** "The system finds relevant text snippets and uses an LLM to synthesize an answer. Notice how the sources show *which* text chunks were used."
- **Subsection 2.4: Visualize Sources (Optional)**
    - Explain this cell shows how you *could* analyze the sources (though the `VectorRAG` class currently doesn't return the 'analysis' dictionary needed for this specific plot).
    - **Presenter action:** Run the cell (it might produce an error or empty plot as 'analysis' isn't returned, which is okay to point out). *Alternatively, modify `VectorRAG` to return source/entity stats, or skip/comment out this cell during presentation.*
    - **Key talking point:** "Visualizing sources can help understand which documents or entity types are most relevant, though our current basic implementation doesn't provide this specific analysis output."

### Section 3: Query-Based RAG (Graph RAG)
- **Goal:** Demonstrate RAG using NL-to-SPARQL translation against a graph database.
- Explain the key components: `pyoxigraph` store, LLM for SPARQL generation, schema hints, and few-shot examples.
- **Subsection 3.1: Initialization Code Cell**
    - Explain the `QueryRAG` class from `src.graph_rag.query_based.py`.
    - It loads TTL files into a `pyoxigraph.Store`.
    - It uses an OpenAI model (`gpt-4o` by default) to translate questions to SPARQL, guided by `reduced_schema.txt` and `few_shot_examples.json`.
    - **Crucially:** Emphasize the need for an `OPENAI_API_KEY` environment variable for this section.
    - **Presenter action:** Run the cell. If the API key is set, observe the TTL loading messages and successful initialization of the `query_rag` instance. If not set, point out the warning message.
    - **Key talking point:** "This system loads the actual RDF graph structure. It needs an OpenAI key because it uses a powerful LLM (GPT-4o) to write SPARQL queries based on the schema and examples we provide."
- **Subsection 3.2: Querying Code Cell**
    - Show the example questions designed to test different SPARQL patterns (COUNT, SELECT+label, Property Path, ASK).
    - Explain the `query_rag.query(question)` method:
        1.  Sends the question, schema, and examples to the LLM to generate SPARQL.
        2.  Executes the generated SPARQL against the `pyoxigraph` store.
        3.  Formats the raw results (potentially using the LLM again) into a natural language answer.
    - **Presenter action:** Run the cell (requires API key). For each question, discuss:
        - The generated SPARQL query (how it reflects the question and schema).
        - The raw results from the graph store.
        - The final formatted answer.
    - **Key talking point:** "Here, the LLM acts as a translator to query the structured graph data directly. Compare the SPARQL and raw results to the final answer. This approach can be very precise if the query is generated correctly."

### Section 5: Understanding the RAG Pipeline(s)
- Briefly summarize the key differences between the two approaches demonstrated:
    - **Vector:** Semantic search on text chunks -> LLM synthesizes answer from retrieved text. Good for finding relevant passages.
    - **Graph (Query):** NL -> SPARQL -> Query graph -> Format results. Good for precise answers based on structured data relationships.
- **Presenter action:** Walk through the bullet points in the markdown cell.
- **Key talking point:** "We've seen two ways to query our BIM data: one using similarity search on text, the other using structured queries on a graph."

### Section 6: Next Steps
- Briefly cover potential extensions for both approaches (better prompts, different models, hybrid methods).
- **Presenter action:** Briefly show the suggestions in the markdown cell.
- **Key talking point:** "These are just starting points. RAG is an active area, and many improvements and combinations are possible."

### Conclusion
- Recap the two RAG methods demonstrated.
- Reiterate the trade-offs (Vector: broader semantic search; Graph/Query: precise structured queries).
- Thank attendees and open for questions.

## Demonstration Tips
- **API Key:** Ensure the `OPENAI_API_KEY` environment variable is set *before* starting the demo if you want Section 3 to run live. Otherwise, be prepared to explain the code without executing it.
- **Caching:** The `QueryRAG` class loads TTLs on init. Re-running the init cell will reload, which might take time.
- **Focus:** Keep the focus on comparing the *process* and *outputs* of the two RAG methods.
- **Simplicity:** The removal of the interactive widget keeps the notebook focused on the core comparison.
