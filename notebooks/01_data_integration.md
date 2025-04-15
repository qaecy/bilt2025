# Data Integration Notebook - Presenter Notes

## Overview
This notebook demonstrates the process of converting raw building information data (IFC files and PDF documents) into RDF graph format for further processing with Graph RAG. It serves as the first part of a lab-presentation where students will learn to preprocess architectural building data for knowledge graph applications.

## Presentation Flow
Follow this guide to present the notebook from start to finish in a natural, logical flow. Each section builds on the previous one, creating a complete data processing pipeline.

### Introduction (Title Slide)
- Welcome attendees and introduce the topic of data integration for building information modeling
- Explain that this is part 1 of a two-part lab on Graph RAG
- Highlight the goal: converting building data into a format suitable for knowledge graphs
- **Presenter action:** Show the title slide and overview bullet points

### Section 0: Setup
- Explain that the setup cell automatically configures the environment for both Colab and local execution
- Point out how it detects the environment and installs dependencies from requirements.txt
- **Presenter action:** Run the cell and verify successful setup
- **Key talking point:** "This setup ensures we can run the notebook anywhere with minimal configuration"

### Prerequisites Section
- Note that all required packages are documented in requirements.txt and installed automatically
- Highlight the key libraries: data processing (pandas, rdflib), IFC processing (ifcopenshell), and PDF processing (pymupdf)
- **Presenter action:** Briefly show the data flow diagram to give an overview of the process
- **Key talking point:** "This single dependency source ensures consistent environments for everyone"

### Section 1: Import Libraries
- Briefly explain the core libraries being imported and their purposes
- Point out the different handling for Colab vs. local environments
- **Presenter action:** Run the cell and confirm successful imports
- **Key talking point:** "These libraries give us the tools to process both geometric and textual building data"

### Section 2: Define Data Paths
- Explain the consistent approach to finding paths across environments
- Point out the directory structure for inputs and outputs
- **Presenter action:** Run the cell and note the configured paths
- **Key talking point:** "This unified approach handles paths consistently regardless of where we're running"

### Section 3: Data Exploration
- Show the available data files and their characteristics
- Highlight the difference between IFC and PDF files
- **Presenter action:** Run the cell and observe the DataFrame output
- **Key talking point:** "We have two complementary data sources: IFC files for geometry and PDF files for specifications"

### Section 4: IFC Processing
- Explain the unified processing function that works for both IFC and PDF files
- Point out the caching mechanism for efficient processing
- **Presenter action:** 
  - Run the cell and observe the function definition and processing results
  - If time permits, process one small file by setting force_reprocess=True
- **Key talking point:** "Our modular approach lets us process different file types with the same workflow"

### Section 5: PDF Processing
- Explain how we're reusing the same function for PDF files
- Highlight how the function handles different converter signatures
- **Presenter action:** Run the cell and observe similar output to IFC processing
- **Key talking point:** "The same function handles both file types, making our code more maintainable"

### Section 6: Results Analysis
- Show the comparison metrics between IFC and PDF data
- Explain the analyze_graph helper function that processes each TTL file
- Point out the visualizations and the ratio comparison
- **Presenter action:** Run the cell and let everyone observe the charts and table
- **Key talking point:** "This analysis quantifies how much richer IFC data is compared to PDFs"

### Section 7: Triple Inspection
- Explain what triples are and how they form the knowledge graph
- Point out the more elegant list comprehension for sample selection
- **Presenter action:** Run the cell and walk through a few example triples
- **Key talking point:** "These triples form the foundation of our knowledge graph"

### Section 8: Entity-Centric Embedding Generation
- Explain the concept of entity-centric embeddings and their importance for Graph RAG
- Highlight how we group triples by subject to create comprehensive entity representations
- Point out the use of sentence-transformers for generating embeddings
- **Presenter action:** 
  - Run the cell and show the embedding files generated
  - Display the sample embedding structure and entity text
- **Key talking point:** "These embeddings will enable semantic search and retrieval in our Graph RAG system"

### Section 9: Summary
- Recap what we've accomplished in a concise way
- Highlight the key metrics in the summary table
- **Presenter action:** Run the cell to display the summary DataFrame
- **Key talking point:** "We've successfully transformed building data into a format ready for Graph RAG"

### Section 10: Troubleshooting
- Point out the streamlined troubleshooting section with problem/solution format
- Emphasize that most issues relate to environment setup, memory, or file structure
- **Presenter action:** Briefly scroll through this section
- **Key talking point:** "These common solutions will help resolve most issues you might encounter"

## Demonstration Tips

### Timing
- The complete presentation should take 45-60 minutes
- Allocate approximately:
  - 5 minutes for introduction and setup
  - 10 minutes for data exploration
  - 15 minutes for processing (less if using cached results)
  - 15 minutes for analysis
  - 5-10 minutes for summary and questions

### Practical Suggestions
- **Before the lab:** Pre-process all files and keep the results for caching
- **During the lab:** Set `force_reprocess=True` for only one small file to demonstrate processing
- **For interactivity:** Ask attendees to predict the IFC:PDF ratio before revealing
- **Emphasize code quality:** Point out how the notebook uses modular functions and consistent patterns

### Technical Requirements
- Ensure you have at least 8GB RAM for processing IFC files
- Have good internet if running in Colab
- Pre-install all required packages from requirements.txt
- Have a backup of pre-processed files if possible

## Transition to Next Lab
End the presentation by emphasizing how this processed data will form the foundation for the Graph RAG lab, where attendees will:
1. Load the TTL files into a unified knowledge graph
2. Create embeddings for graph entities
3. Implement semantic search over the building data
4. Use the graph structure to enhance query results and provide context for LLMs