# BiLT 2025 - Graph RAG in AEC Workshop

This repository contains materials for the Graph RAG in AEC workshop, demonstrating how to integrate BIM data using knowledge graphs and implement RAG (Retrieval-Augmented Generation) for intelligent querying.

## Quick Start

The easiest way to get started is using Google Colab:

[![Open data integration notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qaecy/built2025/blob/main/notebooks/01_data_integration.ipynb)
[![Open RAG querying notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qaecy/built2025/blob/main/notebooks/02_rag_querying.ipynb)

# **Prerequisites**

In order to run the example code you will need to get an API key from OpenAI through the below steps:

1. Sign Up or Log In
   * Go to [https://platform.openai.com/signup](https://platform.openai.com/signup).
   * Sign up with your email, Google, or Microsoft account.
   * If you already have an account, log in at [https://platform.openai.com/login](https://platform.openai.com/login).
1. Verify Your Email & Identity
   * After signing up, OpenAI will send you a verification email.
   * Complete any additional identity verification if prompted (e.g., phone number).
1. Set Up Billing (if needed)
   * Go to [https://platform.openai.com/account/billing](https://platform.openai.com/account/billing).
   * Add a payment method to unlock usage beyond the free trial (if available).
1. Get an API Key
   * Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).
   * Click **“Create new secret key”**.
   * Give it a name (optional) and generate the key.
   * **Copy it immediately** – you won’t be able to view it again later.
1. Apply the API Key
   * Open the Google Colab Notebook
   * Go to the **Secrets** Tab in the left side menu
   * Click **“Add new secret”**.


## Local Installation

If you prefer to run locally:

```bash
git clone https://github.com/qaecy/built2025.git
cd built2025
pip install -e .
```

## Workshop Structure

1. **Data Integration** ([Notebook 1](notebooks/01_data_integration.ipynb))

   - Converting IFC to Knowledge Graphs
   - Integrating unstructured data
   - Graph visualization

2. **RAG Querying** ([Notebook 2](notebooks/02_rag_querying.ipynb))
   - **Vector-Based RAG**: Querying using semantic search over text embeddings.
   - **Query-Based RAG**: Translating natural language to SPARQL queries for graph databases.

## Requirements

- Python 3.8+
- Check requirements.txt for more details

## Session

# Introduction to Graph RAG in AEC (15 minutes)

- BIM Data Challenges: An overview of the current challenges in managing and extracting insights from Building Information Models (BIM). Highlighting the issues of data fragmentation, where critical information is scattered across various formats like IFC files, Excel spreadsheets, and PDF documents. Discuss how this fragmentation hinders efficient workflows and decision-making.
- Introducing Graph RAG: Explanation of Graph Retrieval-Augmented Generation (RAG) and how it can address these challenges. Emphasis on the power of knowledge graphs in representing complex relationships within BIM data and how RAG leverages Large Language Models (LLMs) to bridge the gap between structured and unstructured information.
- Knowledge Graphs and LLMs: A brief introduction to knowledge graphs, explaining how they store data as entities and relationships. Also, a high-level overview of LLMs and their ability to process natural language queries. Explanation of how the combination of these technologies enables more intuitive and powerful data exploration.

# Hands-on: Data Integration (45 minutes)

- Setup: How are people working with our lab content ( https://github.com/qaecy/built2025.git )?
- IFC to Knowledge Graph Conversion: Guide participants through the process of converting an Industry Foundation Classes (IFC) model into a knowledge graph. Explanation of the steps involved in extracting entities and relationships from the IFC data and representing them in a graph database. Provision of practical examples and tools for this conversion.
- Integrating Unstructured Data: Demonstration of how to integrate data from Excel spreadsheets and PDF documents into the knowledge graph. This may involve using techniques like Optical Character Recognition (OCR) to extract text from PDFs and then linking this information to the relevant entities in the graph.
- Visualizing the Graph: Show participants how to visualize the resulting knowledge graph. This can help them understand the relationships between different data points and how the integrated data provides a more holistic view of the project information.

# Hands-on: RAG Querying (60 minutes)

- Simple RAG: Start with simple RAG queries, where participants use natural language to ask questions about the BIM data. Demonstration of how the system retrieves relevant information from the knowledge graph and presents it to the user.
- NLP-to-SPARQL: Introduce the concept of NLP-to-SPARQL, where natural language queries are automatically translated into SPARQL queries that can be executed against the knowledge graph. This allows for more precise and complex data retrieval.

# Interactive Discussion & Q&A (30 minutes)

- Applications in AEC: Facilitate a discussion on the potential applications of graph RAG in various AEC use cases. Encourage participants to share their ideas and discuss how this technology could transform their workflows.
- Challenges and Limitations: Address the challenges and limitations of implementing graph RAG in real-world projects. This could include issues related to data quality, scalability, and the complexity of setting up the system.
- Future Directions: Explore future directions for research and development in this exciting field. Discuss potential advancements in graph RAG technology and how it could further revolutionize the AEC industry.

## Test Data

### Duplex Apartment Dataset

This workshop uses the well-known "Duplex Apartment" test dataset from buildingSMART, which serves as a "Hello World" example for learning to work with IFC files. The dataset is available [here](https://github.com/buildingsmart-community/Community-Sample-Test-Files/tree/main/IFC%202.3.0.1%20(IFC%202x3)/Duplex%20Apartment).

**Dataset Description:**
The Duplex Apartment is a two-story apartment building that includes a main architectural model along with multiple discipline-specific models (MEP, Electrical, Plumbing) simulating engineering consultants' work on a larger project. For this workshop, we use all available IFC models from the dataset, including the architectural model, MEP, electrical, plumbing, and the rooms and spaces model. Additionally, we incorporate some of the provided Excel spreadsheets and sample PDF documents to demonstrate unstructured data integration into the knowledge graph.

**License:**
The Duplex Apartment dataset is provided under Creative Commons License (CC BY 4.0). Users are free to copy, redistribute, and modify the information, including commercial use. Attribution must credit "BSI (2020) 'Duplex Apartment Test Files,' buildingSMART International" and include the GitHub repository URL. Any modifications to the files must be indicated.

## License

MIT License
