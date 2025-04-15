import os
import json
from pathlib import Path
from tqdm import tqdm
from rdflib import Graph, Literal, RDFS, RDF, BNode, URIRef
from openai import OpenAI, APIError

client = OpenAI()


def get_readable_name(uri_or_literal, labels=None):
    """Extract a readable name from a URI or literal"""
    # If we have labels, use the first one
    if labels and len(labels) > 0:
        return list(labels)[0]

    # If it's a literal, return its string value
    if isinstance(uri_or_literal, Literal):
        return str(uri_or_literal)

    # Handle URIs
    uri = str(uri_or_literal)

    # Remove quotes if present (for string literals)
    if uri.startswith('"') and uri.endswith('"'):
        return uri.strip('"')

    # Try to get the fragment or last path component
    if "#" in uri:
        name = uri.split("#")[-1]
    else:
        name = uri.split("/")[-1]

    # Clean up common URI encodings
    name = name.replace("%24", "$").replace("%20", " ")
    name = name.replace("_", " ").replace("-", " ")

    return name


# Define a set of allowed predicate names (case-insensitive comparison)
# These represent potentially relevant information for common queries
ALLOWED_PREDICATES = {
    # Core Identity & Type
    "label",
    "type",
    "name",
    "mark",
    "reference",
    # Dimensions & Geometry
    "width",
    "height",
    "thickness",
    "length",
    "area",
    "volume",
    "unconnected height",
    # Location & Relationships
    "level",
    "is part of",
    "contains element",
    "connected to",
    "base constraint",
    "top constraint",
    "room bounding",
    # Phasing & Status
    "phase created",
    "status",
    # Classification
    "omniclass number",
    "omniclass title",
    "assembly code",
    "assembly description",
    "classification code",
    "classification description",
    "manufacturer",
    # Common Properties
    "isexternal",
    "firerating",
    "material",
    "finish",
    "color",
    "construction type",
    "function",
    "load bearing",
    "structural usage",
    # Specific Types (Examples)
    "door material",
    "frame material",
    "window material",
    "glazing type",
    "sill height",
    "head height",
    # Raw Text Content (for PDF extractions)
    "chars",
}


def ttl_to_embeddings(ttl_file, output_file, model_name="text-embedding-3-small", batch_size=32):
    """
    Convert a TTL file to entity-centric embeddings using OpenAI and save to a JSON file.

    Args:
        ttl_file: Path to the TTL file
        output_file: Path to save the embeddings JSON file
        model_name: Name of the OpenAI embedding model to use
        batch_size: Batch size for sending requests to OpenAI API (adjust based on testing/limits)
    """
    print(f"Processing {ttl_file} into entity-centric embeddings using OpenAI model: {model_name}")

    # Load the graph
    g = Graph()
    g.parse(ttl_file, format="turtle")

    # Group triples by subject (including types and labels)
    entity_data = {}
    for s, p, o in g:
        s_str = str(s)
        if s_str not in entity_data:
            entity_data[s_str] = {"triples": [], "labels": set(), "types": set()}

        entity_data[s_str]["triples"].append((p, o))
        if p == RDF.type:
            entity_data[s_str]["types"].add(o)
        elif p == RDFS.label or "label" in str(p).lower():
            entity_data[s_str]["labels"].add(str(o))

    # Convert each entity to text selectively
    chunks = []
    processed_blank_nodes = set()  # Keep track of processed PSet blank nodes

    # Define the typical property set predicate URI fragment
    HAS_PROPERTY_SET_URI_PART = "hasPropertySet"

    for entity_uri, data in entity_data.items():
        # Skip processing blank nodes directly if they were already processed as Psets
        if entity_uri in processed_blank_nodes:
            continue

        entity_name = get_readable_name(entity_uri, data["labels"])
        entity_types = data.get("types", set())

        text = ""
        is_content_chunk = any(str(t) == "http://www.w3.org/2011/content#ContentAsText" for t in entity_types)

        if is_content_chunk:
            # Special handling for PDF text chunks: only use 'chars'
            chars_value = ""
            for p, o in data["triples"]:
                if str(p) == "http://www.w3.org/2011/content#chars":
                    chars_value = str(o)
                    break
            if chars_value:
                text = f"Text Content: {chars_value}"
            else:
                text = f"Text Chunk: {entity_name}"

        else:
            # General entity handling (IFC, etc.) - now includes PSet traversal
            text_lines = [f"Entity: {entity_name}"]
            type_names = [get_readable_name(t) for t in entity_types]
            if type_names:
                specific_types = [tn for tn in type_names if tn not in ["Interface", "Building Element Proxy", "Thing"]]
                if specific_types:
                    text_lines.append(f"- Type: {', '.join(specific_types)}")
                elif type_names:
                    text_lines.append(f"- Type: {', '.join(type_names)}")

            included_predicates = set()  # Track predicates included for this entity
            psets_to_process = []  # Store PSet nodes to process

            # First pass: Handle direct properties and identify PSet links
            for p, o in data["triples"]:
                p_str = str(p)
                # Check if this is a link to a Property Set
                if HAS_PROPERTY_SET_URI_PART in p_str:
                    # Check if the object is a node in the graph (URI or BNode)
                    if isinstance(o, (URIRef, BNode)):
                        psets_to_process.append(o)
                        if isinstance(o, BNode):
                            processed_blank_nodes.add(str(o))  # Mark BNode as handled
                    continue  # Don't add the hasPropertySet link itself to text

                # Process direct properties (excluding type/label handled above)
                pred = get_readable_name(p)
                pred_lower = pred.lower()
                if pred_lower not in ALLOWED_PREDICATES or pred_lower in ["type", "label"]:
                    continue

                # --- Start Edit: Handle BNode objects for direct properties ---
                actual_object = o
                if isinstance(o, BNode):
                    # Query for the rdf:value of the BNode
                    value_found = False
                    for _, val_p, val_o in g.triples((o, RDF.value, None)):
                        actual_object = val_o
                        value_found = True
                        break  # Assume only one rdf:value per BNode here
                    # Optional: if no rdf:value found, maybe skip or keep BNode id?
                    # For now, we'll proceed with the BNode id if no value found.
                    # if not value_found: continue # <-- uncomment to skip if no rdf:value

                obj = get_readable_name(actual_object)
                # --- End Edit ---

                if obj.lower() == pred.lower() and isinstance(o, Literal):  # Check original 'o' for Literal type
                    continue

                # --- Start Edit: Skip empty or placeholder values ---
                if not obj or obj.strip() == "":  # Skip empty strings
                    continue
                # Re-check after BNode resolution in case rdf:value was the placeholder
                if obj.lower() == pred.lower():  # Skip if value is just the predicate name
                    continue
                # --- End Edit ---

                if pred_lower in included_predicates:
                    continue

                text_lines.append(f"- {pred}: {obj}")
                included_predicates.add(pred_lower)

            # Second pass: Process properties within identified PSets
            for pset_node in psets_to_process:
                # Query the graph for properties of this PSet node
                for pset_p, pset_o in g.predicate_objects(subject=pset_node):
                    # Skip rdf:type declaration of the PSet itself
                    if pset_p == RDF.type:
                        continue

                    pred = get_readable_name(pset_p)
                    pred_lower = pred.lower()

                    # Check against allowed predicates
                    if pred_lower not in ALLOWED_PREDICATES:
                        continue

                    # --- Start Edit: Handle BNode objects for PSet properties ---
                    actual_pset_object = pset_o
                    if isinstance(pset_o, BNode):
                        # Query for the rdf:value of the BNode
                        value_found = False
                        for _, val_p, val_o in g.triples((pset_o, RDF.value, None)):
                            actual_pset_object = val_o
                            value_found = True
                            break  # Assume only one rdf:value per BNode here
                        # Optional: if no rdf:value found, maybe skip or keep BNode id?
                        # For now, we'll proceed with the BNode id if no value found.
                        # if not value_found: continue # <-- uncomment to skip if no rdf:value

                    obj = get_readable_name(actual_pset_object)
                    # --- End Edit ---

                    # Skip placeholder values
                    if obj.lower() == pred.lower() and isinstance(
                        pset_o, Literal
                    ):  # Check original 'pset_o' for Literal type
                        continue

                    # --- Start Edit: Skip empty or placeholder values ---
                    if not obj or obj.strip() == "":  # Skip empty strings
                        continue
                    # Re-check after BNode resolution in case rdf:value was the placeholder
                    # Note: The previous check `obj.lower() == pred.lower() and isinstance(pset_o, Literal)`
                    # already handles cases where the *Literal* value matches the predicate.
                    # This additional check is mostly for cases where a BNode's rdf:value might be the placeholder,
                    # or if a non-Literal object resolves to the predicate name via get_readable_name.
                    if obj.lower() == pred.lower():  # Skip if value is just the predicate name
                        continue
                    # --- End Edit ---

                    # Skip if already included (avoids duplicates if property exists directly and in PSet)
                    if pred_lower in included_predicates:
                        continue

                    text_lines.append(f"- {pred}: {obj}")
                    included_predicates.add(pred_lower)

            # Combine lines only if there's more than just the entity name
            if len(text_lines) > 1:
                text = "\\n".join(text_lines)
            # else: # Optional: Handle entities with no useful extracted info?
            #    text = f"Entity: {entity_name} (No specific details extracted)"

        # Only add chunk if text was generated
        if text:
            # Replace newline characters with spaces for OpenAI compatibility if needed
            # text = text.replace("\\\\n", " ") # Keep newlines for now, test effectiveness
            chunks.append(
                {
                    "entity": entity_uri,
                    "entity_name": entity_name,
                    "text": text,
                    "source_file": os.path.basename(ttl_file),
                }
            )

    print(f"Generated {len(chunks)} filtered entity chunks (incl. PSet traversal)")

    if not chunks:
        print("No processable chunks generated after filtering.")
        # Optionally save an empty file or skip saving
        # Saving an empty file for consistency:
        metadata = {"model_name": model_name, "source_file": os.path.basename(ttl_file), "entity_count": 0}
        output_data = {"metadata": metadata, "chunks": []}
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f)
        print(f"Saved empty embeddings file to {output_file}")
        return  # Exit the function for this file

    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings in batches using OpenAI API
    print(f"Generating embeddings using OpenAI model: {model_name}...")
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate total batches for tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks", total=total_batches):
        batch_texts = texts[i : i + batch_size]
        try:
            # Use the OpenAI client to create embeddings
            response = client.embeddings.create(input=batch_texts, model=model_name)
            # Extract embeddings from the response object
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except APIError as api_error:
            print(f"OpenAI API Error on batch starting at index {i}: {api_error}")
            # Add None for failed batch embeddings to keep alignment
            all_embeddings.extend([None] * len(batch_texts))
        except Exception as e:
            print(f"Error encoding batch starting at index {i}: {e}")
            # Add None or zeros for failed batch embeddings to keep alignment
            all_embeddings.extend([None] * len(batch_texts))  # Placeholder for failed

    # Filter out chunks where embedding failed before adding to output
    successful_chunks = []
    for i, chunk in enumerate(chunks):
        if i < len(all_embeddings) and all_embeddings[i] is not None:
            # OpenAI returns a list directly, no need for .tolist()
            chunk["embedding"] = all_embeddings[i]
            successful_chunks.append(chunk)
        else:
            print(f"Warning: Skipping chunk for entity {chunk['entity']} due to embedding error.")

    # Create metadata
    metadata = {
        "model_name": model_name,
        "source_file": os.path.basename(ttl_file),
        "entity_count": len(successful_chunks),
    }

    # Save to JSON
    output_data = {"metadata": metadata, "chunks": successful_chunks}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    print(f"Saved {len(successful_chunks)} filtered entity embeddings to {output_file}")


def ttl_files_to_embeddings(input_dir, output_dir, model_name="text-embedding-3-small", force_reprocess=False):
    """
    Convert all TTL files in a directory to entity-centric embeddings using OpenAI.

    Args:
        input_dir: Directory containing TTL files
        output_dir: Directory to save embedding files
        model_name: Name of the OpenAI embedding model to use
        force_reprocess: Whether to reprocess existing embedding files
    """
    # Ensure directories exist
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all TTL files
    ttl_files = list(input_dir.glob("*.ttl"))
    if not ttl_files:
        print(f"No TTL files found in {input_dir}")
        return

    print(f"Found {len(ttl_files)} TTL files to process")

    # Process each file
    for ttl_file in ttl_files:
        output_file = output_dir / f"{ttl_file.stem}_embeddings.json"

        # Skip if output exists and not forced to reprocess
        if output_file.exists() and not force_reprocess:
            print(f"Skipping {ttl_file.name} - embeddings already exist")
            continue

        # try:
        ttl_to_embeddings(ttl_file, output_file, model_name=model_name)
