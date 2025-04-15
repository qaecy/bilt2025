import os
import csv
import fitz
import logging
from pathlib import Path
from multiprocessing import freeze_support
from wtpsplit_lite import SaT
from rdflib.namespace import RDF, XSD, DCTERMS
from rdflib import Graph, Namespace, Literal, URIRef
import requests
from pyshacl import validate
from .common_converter import OntologyManager

# Add SHACL namespace definition
SHACL = Namespace("http://www.w3.org/ns/shacl#")

# Initialize the lightweight SaT model
try:
    chunker = SaT("sat-3l-sm")
    logging.info("Successfully initialized SaT chunker")
except Exception as e:
    logging.error(f"Failed to initialize SaT chunker: {e}")
    chunker = None


def text_from_page(page: fitz.Page):
    """Enhanced text extraction with more metadata and structure"""
    blocks = page.get_textpage().extractDICT(sort=True)["blocks"]
    for block in blocks:
        # Extract block-level information
        block_bbox = block.get("bbox", [])  # Physical coordinates
        block_type = block.get("type", 0)  # 0=text, 1=image, etc.

        for line in block.get("lines", []):
            line_bbox = line.get("bbox", [])
            line_dir = line.get("dir", [])  # Text direction

            for span in line.get("spans", []):
                yield {
                    "text": span.get("text", ""),
                    "font": span.get("font", ""),
                    "size": span.get("size", 0),
                    "color": span.get("color", 0),
                    "bbox": span.get("bbox", []),  # Physical location
                    "block_type": block_type,
                    "block_bbox": block_bbox,
                    "line_bbox": line_bbox,
                    "line_dir": line_dir,
                }


def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """
    Split text using SaT lightweight model and control chunk size
    """
    if not chunker:
        # Fallback to simple splitting if chunker failed to initialize
        return [text]

    try:
        # Use SaT to split into sentences
        sentences = chunker.split(text)

        # Combine sentences into chunks of reasonable size
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    except Exception as e:
        logging.error(f"Error in text chunking: {e}")
        return [text]  # Fallback to returning the entire text as one chunk


def pdf_text_to_chunks(filename: str):
    pdf = fitz.open(filename)
    for i, page in enumerate(pdf):
        text = " ".join(text_from_page(page))
        if len(text):
            for j, chunk in enumerate(chunk_text(text)):
                yield [filename, i, j, chunk]


def pdf_to_csv(*pdf_filenames: tuple[str], csv_filename: str):
    with open(csv_filename, "w") as file:
        writer = csv.writer(file, delimiter=",", quoting=csv.QUOTE_ALL)
        writer.writerow(["filename", "page_index", "chunk_index", "chunk"])
        for pdf_filename in pdf_filenames:
            writer.writerows(pdf_text_to_chunks(pdf_filename))


def files_from_dir(path: str, extension: str | None = None):
    for sub_dir, _, files in os.walk(path):
        for file_name in files:
            if extension is None or file_name.lower().endswith(extension):
                absolute_file_path = os.path.join(path, sub_dir, file_name)
                yield absolute_file_path


def pdf_files_from_dir(path: str):
    return files_from_dir(path, ".pdf")


# Update ONTOLOGY_URLS without CO
ONTOLOGY_URLS = {
    "fabio": ("http://purl.org/spar/fabio/", "xml"),
    "prov": ("http://www.w3.org/ns/prov#", "turtle"),
    "cnt": ("http://www.w3.org/2011/content#", "xml"),
    "dcterms": ("http://purl.org/dc/terms/", "turtle"),
}


def fetch_and_store_ontologies(base_dir: Path) -> dict:
    """Fetch ontologies and store them locally"""
    ontology_dir = base_dir / "data" / "ontologies"
    ontology_dir.mkdir(parents=True, exist_ok=True)

    ontology_graphs = {}

    for name, (url, format_hint) in ONTOLOGY_URLS.items():
        output_path = ontology_dir / f"{name}.ttl"

        # First try to load from local file
        if output_path.exists():
            try:
                g = Graph()
                g.parse(str(output_path), format="turtle")
                ontology_graphs[name] = g
                logging.info(f"Loaded existing {name} ontology from {output_path}")
                continue
            except Exception as e:
                logging.warning(f"Failed to load local {name} ontology: {e}")

        # If local file doesn't exist or is invalid, try to fetch
        try:
            logging.info(f"Fetching ontology: {name}")
            headers = {
                "Accept": "application/rdf+xml, text/turtle",
                "User-Agent": "Built2025/1.0",
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            g = Graph()
            try:
                g.parse(data=response.text, format=format_hint)
            except Exception:
                # Try alternative formats
                for fmt in ["xml", "turtle", "n3"]:
                    try:
                        g.parse(data=response.text, format=fmt)
                        break
                    except:
                        continue

            g.serialize(destination=str(output_path), format="turtle")
            ontology_graphs[name] = g
            logging.info(f"Successfully stored {name} ontology")

        except Exception as e:
            logging.error(f"Error fetching ontology {name}: {e}")

    return ontology_graphs


def validate_graph(g: Graph, shapes_file: Path) -> tuple[bool, Graph, str]:
    """Validate the generated RDF against SHACL shapes"""
    if not shapes_file.exists():
        logging.warning("No shapes file found, skipping validation")
        return True, None, "No validation performed"

    try:
        shapes_graph = Graph()
        shapes_graph.parse(str(shapes_file), format="turtle")

        conforms, results_graph, results_text = validate(
            g, shacl_graph=shapes_graph, inference="rdfs", abort_on_first=False
        )

        if not conforms:
            logging.warning(f"Validation failed:\n{results_text}")

        return conforms, results_graph, results_text
    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return True, None, "Validation failed but continuing"


def pdf_to_ttl(pdf_filenames: list[str], output_file: str):
    g = Graph()

    # Create ontology manager with explicitly needed ontologies
    ontology_manager = OntologyManager(
        ontologies={
            "fabio": ("http://purl.org/spar/fabio/", "xml"),
            "prov": ("http://www.w3.org/ns/prov#", "turtle"),
            "cnt": ("http://www.w3.org/2011/content#", "xml"),
            "dcterms": ("http://purl.org/dc/terms/", "turtle"),
        }
    )

    # Bind namespaces
    for prefix in ["fabio", "dcterms", "prov", "cnt"]:
        ns = ontology_manager.get_namespace(prefix)
        if ns:
            g.bind(prefix, ns)

    # Custom PDO namespace (not in standard ontologies)
    PDO = Namespace("http://www.semanticweb.org/ontologies/pdo#")
    g.bind("pdo", PDO)

    # Create proper URIRef for index property
    PDO_INDEX = URIRef(str(PDO) + "index")

    try:
        for pdf_filename in pdf_filenames:
            pdf = fitz.open(pdf_filename)
            doc_uri = URIRef(f"urn:document:{os.path.basename(pdf_filename)}")

            # Document metadata
            g.add((doc_uri, RDF.type, ontology_manager.get_namespace("fabio").DigitalDocument))
            g.add((doc_uri, ontology_manager.get_namespace("dcterms").hasFormat, Literal("application/pdf")))
            g.add((doc_uri, ontology_manager.get_namespace("dcterms").title, Literal(os.path.basename(pdf_filename))))

            # Extract PDF metadata
            metadata = pdf.metadata
            if metadata:
                if metadata.get("title"):
                    g.add((doc_uri, ontology_manager.get_namespace("dcterms").title, Literal(metadata["title"])))
                if metadata.get("author"):
                    g.add((doc_uri, ontology_manager.get_namespace("dcterms").creator, Literal(metadata["author"])))
                if metadata.get("subject"):
                    g.add((doc_uri, ontology_manager.get_namespace("dcterms").subject, Literal(metadata["subject"])))
                if metadata.get("keywords"):
                    g.add((doc_uri, ontology_manager.get_namespace("dcterms").subject, Literal(metadata["keywords"])))
                if metadata.get("producer"):
                    g.add(
                        (doc_uri, ontology_manager.get_namespace("prov").wasGeneratedBy, Literal(metadata["producer"]))
                    )
                if metadata.get("creationDate"):
                    g.add(
                        (doc_uri, ontology_manager.get_namespace("dcterms").created, Literal(metadata["creationDate"]))
                    )
                if metadata.get("modDate"):
                    g.add((doc_uri, ontology_manager.get_namespace("dcterms").modified, Literal(metadata["modDate"])))

            # Document structure
            g.add((doc_uri, PDO.pageCount, Literal(len(pdf), datatype=XSD.integer)))

            for page_num, page in enumerate(pdf):
                page_uri = URIRef(f"{doc_uri}/page_{page_num}")

                # Page metadata
                g.add((page_uri, RDF.type, ontology_manager.get_namespace("fabio").Page))
                g.add((page_uri, PDO_INDEX, Literal(page_num, datatype=XSD.integer)))
                g.add((page_uri, ontology_manager.get_namespace("dcterms").isPartOf, doc_uri))
                g.add((page_uri, PDO.width, Literal(page.rect.width, datatype=XSD.float)))
                g.add((page_uri, PDO.height, Literal(page.rect.height, datatype=XSD.float)))
                g.add((page_uri, PDO.rotation, Literal(page.rotation, datatype=XSD.integer)))

                # Extract text with rich metadata
                text_blocks = list(text_from_page(page))
                current_text = ""
                current_metadata = {}
                chunk_num = 0  # Initialize chunk counter for this page

                for block in text_blocks:
                    if len(current_text) + len(block["text"]) > 500:
                        # Create chunk with accumulated text
                        chunk_uri = URIRef(f"{page_uri}/chunk_{chunk_num}")
                        g.add((chunk_uri, RDF.type, ontology_manager.get_namespace("cnt").ContentAsText))
                        g.add((chunk_uri, ontology_manager.get_namespace("cnt").chars, Literal(current_text)))
                        g.add((chunk_uri, ontology_manager.get_namespace("dcterms").isPartOf, page_uri))
                        g.add((chunk_uri, PDO_INDEX, Literal(chunk_num, datatype=XSD.integer)))

                        # Add formatting metadata
                        if current_metadata.get("font"):
                            g.add((chunk_uri, PDO.font, Literal(current_metadata["font"])))
                        if current_metadata.get("size"):
                            g.add((chunk_uri, PDO.fontSize, Literal(current_metadata["size"], datatype=XSD.float)))
                        if current_metadata.get("color"):
                            g.add((chunk_uri, PDO.color, Literal(current_metadata["color"])))
                        if current_metadata.get("bbox"):
                            bbox = current_metadata["bbox"]
                            g.add((chunk_uri, PDO.boundingBox, Literal(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")))

                        chunk_num += 1  # Increment chunk counter
                        current_text = block["text"]
                        current_metadata = block
                    else:
                        current_text += " " + block["text"]
                        # Update metadata for dominant properties
                        if block.get("size", 0) > current_metadata.get("size", 0):
                            current_metadata.update(block)

                # Don't forget the last chunk
                if current_text:
                    chunk_uri = URIRef(f"{page_uri}/chunk_{chunk_num}")
                    g.add((chunk_uri, RDF.type, ontology_manager.get_namespace("cnt").ContentAsText))
                    g.add((chunk_uri, ontology_manager.get_namespace("cnt").chars, Literal(current_text)))
                    g.add((chunk_uri, ontology_manager.get_namespace("dcterms").isPartOf, page_uri))
                    g.add((chunk_uri, PDO_INDEX, Literal(chunk_num, datatype=XSD.integer)))

                    # Add final chunk metadata
                    if current_metadata.get("font"):
                        g.add((chunk_uri, PDO.font, Literal(current_metadata["font"])))
                    if current_metadata.get("size"):
                        g.add((chunk_uri, PDO.fontSize, Literal(current_metadata["size"], datatype=XSD.float)))
                    if current_metadata.get("color"):
                        g.add((chunk_uri, PDO.color, Literal(current_metadata["color"])))
                    if current_metadata.get("bbox"):
                        bbox = current_metadata["bbox"]
                        g.add((chunk_uri, PDO.boundingBox, Literal(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")))

        # Validate before serializing
        shapes_file = Path(__file__).parent / "pdf_converter_shapes.ttl"
        if shapes_file.exists():
            conforms, _, results_text = validate_graph(g, shapes_file)
            if not conforms:
                logging.warning("Graph validation failed, but continuing with serialization")

        g.serialize(destination=output_file, format="turtle")
        logging.info(f"Successfully created: {output_file}")

    except Exception as e:
        logging.error(f"Error processing PDF to TTL: {e}")
        raise


if __name__ == "__main__":
    freeze_support()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Fetch and store ontologies
    base_dir = Path(__file__).parent.parent.parent
    ontology_graphs = fetch_and_store_ontologies(base_dir)

    root_dir = Path(__file__).parent.parent.parent / "data"
    project_names = ["buildingsmart_duplex", "buildingsmart_dental", "buildingsmart_schependomlaan"]

    for project_name in project_names:
        raw_dir = root_dir / "raw" / project_name
        graph_dir = root_dir / "graph" / project_name
        graph_dir.mkdir(parents=True, exist_ok=True)

        for pdf_filepath in raw_dir.glob("*.pdf"):
            ttl_filepath = graph_dir / pdf_filepath.name.replace(".pdf", ".ttl")
            print(f"\nProcessing {pdf_filepath.name}")
            pdf_to_ttl([pdf_filepath], ttl_filepath)
