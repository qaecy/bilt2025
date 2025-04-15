import json
import pyoxigraph as ox
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryRAG:
    """RAG system using a pyoxigraph graph store and NL-to-SPARQL translation."""

    def __init__(
        self,
        ttl_files: List[Path],
        schema_file: Path,
        examples_file: Path,
        llm_model: str = "gpt-4o",  # Using a more capable model for SPARQL generation
    ):
        """Initialize RAG system with TTL files, schema, examples, and LLM.

        Args:
            ttl_files: List of paths to TTL files containing RDF triples.
            schema_file: Path to the text file containing the boiled-down schema.
            examples_file: Path to the JSON file containing few-shot examples.
            llm_model: Name of the OpenAI chat model for NL-to-SPARQL.
        """
        self.store = ox.Store()
        self._load_graph(ttl_files)

        # Load schema and examples for the prompt context
        self.schema_context = self._load_text_file(schema_file)
        self.few_shot_examples = self._load_json_file(examples_file)

        if not self.schema_context:
            print("Warning: Schema context could not be loaded. SPARQL generation may be inaccurate.")
        if not self.few_shot_examples:
            print("Warning: Few-shot examples could not be loaded. SPARQL generation may be inaccurate.")

        # Initialize LLM chain for SPARQL generation
        # Assumes OPENAI_API_KEY is set in environment
        try:
            # Using a more capable model like gpt-4o is recommended for reliable SPARQL generation
            self.llm = ChatOpenAI(model_name=llm_model, temperature=0.0)  # Low temp for deterministic SPARQL
            self.output_parser = StrOutputParser()
            self.prompt_template = self._build_prompt_template()
            self.chain = self.prompt_template | self.llm | self.output_parser
        except Exception as e:
            print(f"Error initializing LLM chain: {e}. Ensure OPENAI_API_KEY is set and dependencies are installed.")
            self.llm = None
            self.chain = None

        print(f"GraphRAG initialized with {len(ttl_files)} TTL files.")
        print(f"Total triples loaded: {len(self.store)}")

    def _load_text_file(self, file_path: Path) -> Optional[str]:
        """Load content from a text file."""
        try:
            if file_path.is_file():
                with open(file_path, "r") as f:
                    return f.read()
            else:
                print(f"Warning: File not found: {file_path}")
                return None
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return None

    def _load_json_file(self, file_path: Path) -> Optional[List[Dict]]:
        """Load content from a JSON file."""
        try:
            if file_path.is_file():
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                print(f"Warning: File not found: {file_path}")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {file_path}: {e}")
            return None
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build the prompt template for the LLM chain."""
        examples_str = ""
        if self.few_shot_examples:
            for example in self.few_shot_examples:
                # Escape literal curly braces in the SPARQL example (`{`, `}`)
                # because they are used by f-strings for formatting.
                escaped_sparql = example["sparql"].replace("{", "{{").replace("}", "}}")
                examples_str += (
                    f"Question: {example['question']}\\nSPARQL Query:\\n```sparql\\n{escaped_sparql}\\n```\\n\\n"
                )

        # Construct the prompt template string.
        # Note: {{question}} uses double braces because it IS the intended input variable for the LangChain prompt template.
        # {self.schema_context} and {examples_str} use single braces because they are
        # Python variables being formatted into the template string itself using an f-string.
        template = f"""
You are an expert SPARQL query writer specializing in IFC-based RDF graphs (using ifcOWL conventions).
Your task is to translate the user's natural language question into a valid SPARQL query based on the provided schema information and examples.

Follow these instructions carefully:
1. Use ONLY the prefixes, classes, and properties defined in the 'Schema Information' section below.
2. Pay close attention to the patterns shown in the 'Query Examples' section, especially for accessing property values via property sets (often using `ifc:hasPropertySet`).
3. Generate ONLY the SPARQL query. Do not include any explanations, introductions, or markdown formatting around the query itself.
4. If the question cannot be answered using the provided schema, return the text 'QUERY_NOT_POSSIBLE'.
5. Ensure all necessary prefixes are included in the generated query. **CRITICAL: You MUST include standard prefixes (`rdf:`, `rdfs:`, `ifc:`, `xsd:`) in the query if any terms from these namespaces are used in the query body. Check your generated query for terms like `rdf:type` or `rdfs:label` and ensure their prefixes are defined.**
6. Assume standard `rdf:value` is used to access literal values within property value nodes unless the schema or examples indicate otherwise.

Schema Information:
---
{self.schema_context or 'Schema information not available.'}
---

Query Examples:
---
{examples_str or 'No examples available.'}
---

User Question: {{question}}

SPARQL Query:
```sparql
"""
        return ChatPromptTemplate.from_template(template)

    def _load_graph(self, ttl_files: List[Path]):
        """Load RDF triples from a list of TTL files into the pyoxigraph store."""
        print("Loading TTL files into graph store...")
        count = 0
        for file_path in ttl_files:
            if not file_path.is_file():
                print(f"Warning: Skipping non-existent file {file_path}")
                continue

            try:
                # Define base IRI for resolving relative IRIs within the TTL file, if any
                base_iri = f"file://{file_path.resolve()}/"
                with open(file_path, "rb") as f:
                    self.store.load(f, "text/turtle", base_iri=base_iri)
                print(f"Successfully loaded: {file_path.name}")
                count += 1
            except Exception as e:
                # Provide specific error feedback if possible
                print(f"Warning: Error loading file {file_path}: {e}")
                if "Unrecognized predicate" in str(e) or "Unexpected token" in str(e):
                    print("  Hint: This might be due to incorrect RDF syntax or undefined terms in the TTL file.")
                elif "relative IRI" in str(e):
                    print("  Hint: Ensure a proper base IRI is set or avoid relative IRIs in the TTL file.")
                continue
        print(f"Finished loading {count} files.")

    def query(self, question: str) -> Dict[str, Any]:
        """Answer a question by converting it to SPARQL and querying the graph."""
        if not self.chain:
            return {
                "answer": "LLM chain not initialized. Cannot process query.",
                "sparql_query": None,
                "raw_results": None,
            }

        # 1. Generate SPARQL query using the LLM chain
        sparql_query = self._generate_sparql(question)

        if not sparql_query or sparql_query == "QUERY_NOT_POSSIBLE":
            query_failure_reason = (
                "Failed to generate SPARQL query."
                if not sparql_query
                else "Query generation deemed not possible based on schema."
            )
            return {"answer": query_failure_reason, "sparql_query": sparql_query, "raw_results": None}

        # 2. Execute SPARQL query against the graph store
        raw_results = self._execute_sparql(sparql_query)

        # 3. Format results into a natural language answer
        answer = self._format_results(question, raw_results)

        return {
            "answer": answer,
            "sparql_query": sparql_query,
            "raw_results": raw_results,  # Include raw results for debugging/transparency
        }

    def _generate_sparql(self, question: str) -> Optional[str]:
        """Use LLM chain to generate a SPARQL query from a natural language question."""
        print(f"Generating SPARQL for question: '{question}'")
        if not self.chain:
            print("Error: LLM chain is not initialized.")
            return None
        try:
            response = self.chain.invoke({"question": question})

            # Clean up the response - remove potential markdown backticks added by the LLM
            if response.startswith("```sparql"):
                response = response[len("```sparql") :].strip()
            if response.endswith("```"):
                response = response[: -len("```")].strip()

            if response == "QUERY_NOT_POSSIBLE":
                print("LLM determined query is not possible based on schema/examples.")
                return "QUERY_NOT_POSSIBLE"

            # Basic validation: Check if it looks like a standard SPARQL query type
            if (
                "SELECT" not in response.upper()
                and "ASK" not in response.upper()
                and "CONSTRUCT" not in response.upper()
                and "DESCRIBE" not in response.upper()
            ):
                print(f"Warning: Generated text does not look like a valid SPARQL query:\\n{response}")
                # Allow potentially malformed queries for now, execution will likely fail
                pass

            print(f"Generated SPARQL (raw):\\n{response}")
            return response

        except Exception as e:
            print(f"Error during SPARQL generation: {e}")
            return None

    def _execute_sparql(self, sparql_query: str) -> Optional[List[Dict[str, str]]]:
        """Execute a SPARQL query against the pyoxigraph store."""

        # --- Add missing prefixes programmatically --- #
        # Check if common prefixes are used in the query body but not declared,
        # and add them to prevent SPARQL parsing errors.
        required_prefixes = {
            "rdf:": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            "rdfs:": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
            "ifc:": "PREFIX ifc: <http://ifcowl.openbimstandards.org/IFC2X3#>",
            "xsd:": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
        }
        prefixes_to_add = []
        defined_prefixes = [
            line.split()[1] for line in sparql_query.splitlines() if line.strip().upper().startswith("PREFIX")
        ]

        for prefix, declaration in required_prefixes.items():
            if prefix in sparql_query and prefix not in defined_prefixes:
                print(f"Note: Adding missing prefix declaration: {declaration}")
                prefixes_to_add.append(declaration)

        if prefixes_to_add:
            sparql_query = "\\n".join(prefixes_to_add) + "\\n" + sparql_query
        # --- End prefix addition --- #

        print(f"Executing SPARQL (final):\\n{sparql_query}")
        try:
            results = []
            query_results = self.store.query(sparql_query)

            # --- Handle different pyoxigraph result types ---

            # Handle ASK query result FIRST - This is critical as its type is distinct.
            # Check for the specific QueryBoolean type from pyoxigraph using its type name (as it's not directly exposed).
            if type(query_results).__name__ == "QueryBoolean":
                print("Query returned a QueryBoolean result (ASK query).")
                # Process the boolean result.
                boolean_value_str = str(query_results).lower()
                boolean_value = boolean_value_str == "true"  # Explicitly check for 'true'
                print(f"Extracted boolean value: {boolean_value}")
                return [{"boolean_result": str(boolean_value).lower()}]

            # --- If not QueryBoolean, proceed assuming SELECT or other iterable results ---
            print("Query returned non-boolean results (likely SELECT/CONSTRUCT/DESCRIBE). Processing...")

            # Get variable names from the iterator object (for SELECT queries)
            try:
                # Ensure query_results is actually iterable before accessing .variables
                iter(query_results)
                variables = [v.value for v in query_results.variables]
                print(f"Query variables: {variables}")
            except TypeError:
                # This path should ideally not be hit if the boolean check is correct.
                print("Error: Query result object is not iterable and not boolean.")
                raise ValueError("Query result from pyoxigraph is neither boolean nor iterable.") from None
            except AttributeError:
                # This might happen if the query fails early or is not SELECT/ASK.
                print(
                    "Warning: Could not retrieve variables from query results. Query might have failed or is not SELECT/ASK."
                )
                variables = []  # Proceed without known variables, might lead to issues

            # Iterate through the solutions (rows) in the results
            for solution in query_results:
                result_dict = {}
                if not variables:
                    # If variables weren't retrieved (e.g., CONSTRUCT/DESCRIBE), try a generic approach.
                    try:
                        result_dict["_solution"] = str(solution)  # This might just show the raw triple string
                    except Exception:
                        result_dict["_error"] = "Failed to process solution"
                else:
                    # For SELECT queries, process results by variable name
                    for var_name in variables:
                        try:
                            term = solution[var_name]  # Access term by variable name (string)
                            if term is None:
                                continue  # Skip unbound variables in the solution

                            # Process term based on its RDF type (NamedNode, BlankNode, Literal)
                            if isinstance(term, ox.NamedNode):
                                processed_term = f"<{term.value}>"  # URI
                            elif isinstance(term, ox.BlankNode):
                                processed_term = f"_:{term.value}"  # Blank node ID
                            elif isinstance(term, ox.Literal):
                                # Format literals, including datatype and language tag if present
                                if (
                                    term.datatype and term.datatype.value != "http://www.w3.org/2001/XMLSchema#string"
                                ):  # Don't add ^^xsd:string, as it's the default
                                    processed_term = f'"{term.value}"^^<{term.datatype.value}>'
                                elif term.language:
                                    processed_term = f'"{term.value}"@{term.language}'
                                else:
                                    processed_term = f'"{term.value}"'  # Simple literal
                            else:
                                processed_term = str(term)  # Fallback for unknown term types
                            result_dict[var_name] = processed_term
                        except (KeyError, TypeError) as term_error:
                            # Handle cases where variable might not be in solution or indexing fails unexpectedly
                            print(f"Warning: Could not access variable '{var_name}' in solution: {term_error}")
                            result_dict[var_name] = None  # Indicate error or missing value
                if result_dict:  # Only add if we could process something for this solution
                    results.append(result_dict)

            print(f"Query returned {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error executing SPARQL query: {e}")
            # Provide specific feedback for common SPARQL errors
            if "ParseError" in str(e):
                print("  Hint: The generated SPARQL query has syntax errors.")
            elif "QueryEvaluationError" in str(e):
                print(
                    "  Hint: The query is syntactically valid but failed during evaluation (e.g., invalid path, type error)."
                )
            # Add hint for specific pyoxigraph processing issues
            elif "object has no attribute" in str(e) or "is not iterable" in str(e):
                print("  Hint: There was an issue processing the query results format from pyoxigraph.")
            return None

    def _format_results(self, question: str, results: Optional[List[Dict[str, str]]]) -> str:
        """Format SPARQL query results into a natural language answer using an LLM."""
        if results is None:
            return "An error occurred while querying the graph."
        if not results:
            return "No information found in the graph for your question."

        # Handle ASK query result directly (already formatted in _execute_sparql)
        if len(results) == 1 and "boolean_result" in results[0]:
            bool_ans = results[0]["boolean_result"]
            # Simple phrasing for boolean results
            return f"The answer to '{question}' is {bool_ans}."

        # --- Use LLM to format SELECT/other results --- #
        print("Formatting results using LLM...")
        try:
            # Prepare results string for the prompt
            max_results_for_prompt = 10  # Limit number of results shown to LLM to manage context window
            results_str = json.dumps(results[:max_results_for_prompt], indent=2)
            if len(results) > max_results_for_prompt:
                results_str += f"\\n... ({len(results) - max_results_for_prompt} more results truncated)"

            # Create a dedicated prompt for answer synthesis based on query results
            synthesis_prompt = ChatPromptTemplate.from_template(
                f"""You are an assistant tasked with summarizing SPARQL query results into a concise, natural language answer.

                Original Question: {{question}}

                Query Results (JSON format):
                {{results_str}}

                Based *only* on the provided query results, answer the original question clearly and concisely.
                - If the results provide a single, direct answer (like a count or a specific value), state it directly.
                - If the results list multiple items or values for the *same* requested property (e.g., multiple different areas for the same space label), explicitly mention that multiple differing values were found and list them or summarize the range.
                - If the results list different items (e.g., multiple wall labels), summarize them briefly (e.g., "The following items were found: ...").
                - Do not add any information not present in the results.
                - If the results are complex or numerous (more than ~5-10 entries), provide a high-level summary and indicate that more results are available.
                - If the results seem empty or don't answer the question, state that clearly.

                Answer:
                """
            )

            synthesis_chain = synthesis_prompt | self.llm | self.output_parser

            formatted_answer = synthesis_chain.invoke({"question": question, "results_str": results_str})
            return formatted_answer

        except Exception as e:
            print(f"Error during LLM result formatting: {e}")
            # Fallback to basic formatting if LLM synthesis fails
            print("Falling back to basic result formatting.")
            formatted = f"Found {len(results)} results for '{question}' (LLM formatting failed):\\n"
            max_results_to_show = 5  # Basic fallback limit
            for i, res in enumerate(results[:max_results_to_show]):
                res_str = ", ".join([f"{k}: {v}" for k, v in res.items()])
                formatted += f"- Result {i+1}: {{ {res_str} }}\\n"
            if len(results) > max_results_to_show:
                formatted += f"- ... ({len(results) - max_results_to_show} more results)\\n"
            return formatted
