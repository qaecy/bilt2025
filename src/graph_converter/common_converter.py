import logging
import requests
from pathlib import Path
from rdflib import Graph, Namespace
from typing import Dict, Optional, Tuple


class OntologyManager:
    """Centralized ontology management for fetching and caching ontologies."""

    def __init__(self, cache_dir: str = "data/ontologies", ontologies: Dict[str, Tuple[str, str]] | None = None):
        """
        Initialize the ontology manager.

        Args:
            cache_dir: Directory to store cached ontologies
            ontologies: Dictionary of ontologies to manage
                       Format: {"prefix": ("url", "format")}
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if ontologies is None:
            raise ValueError("Ontologies dictionary must be provided")

        self.ontologies = ontologies
        self.known_ontologies: Dict[str, Graph] = {}
        self.failed_ontologies = set()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def get_ontology(self, prefix: str) -> Optional[Graph]:
        """
        Get an ontology by its prefix, either from cache or by fetching.

        Args:
            prefix: The ontology prefix (e.g., "bot", "prov")

        Returns:
            The ontology graph or None if not available
        """
        if prefix not in self.ontologies:
            self.logger.warning(f"Unknown ontology prefix: {prefix}")
            return None

        if prefix in self.known_ontologies:
            return self.known_ontologies[prefix]

        if prefix in self.failed_ontologies:
            return None

        url, format_hint = self.ontologies[prefix]
        cache_path = self.cache_dir / f"{prefix}.ttl"

        # Try loading from cache first
        if cache_path.exists():
            try:
                g = Graph()
                g.parse(str(cache_path), format="turtle")
                self.known_ontologies[prefix] = g
                return g
            except Exception as e:
                self.logger.warning(f"Failed to load cached ontology {prefix}: {e}")
                cache_path.unlink(missing_ok=True)

        # Fetch if not in cache
        try:
            headers = {
                "Accept": "application/rdf+xml, text/turtle",
                "User-Agent": "OntologyManager/1.0",
            }

            response = requests.get(url, headers=headers, timeout=2)
            response.raise_for_status()

            g = Graph()

            # Try specified format first
            try:
                g.parse(data=response.text, format=format_hint)
            except Exception:
                # Try alternative formats if specified format fails
                for fmt in ["turtle", "xml", "n3"]:
                    try:
                        g.parse(data=response.text, format=fmt)
                        break
                    except Exception:
                        continue

            # Cache the successfully parsed ontology
            g.serialize(destination=str(cache_path), format="turtle")
            self.known_ontologies[prefix] = g
            self.logger.info(f"Successfully fetched and cached ontology: {prefix}")
            return g

        except Exception as e:
            self.logger.error(f"Failed to fetch ontology {prefix}: {e}")
            self.failed_ontologies.add(prefix)
            return None

    def get_namespace(self, prefix: str) -> Optional[Namespace]:
        """Get the namespace for a given prefix."""
        if prefix in self.ontologies:
            url, _ = self.ontologies[prefix]
            return Namespace(url)
        return None

    def get_all_ontologies(self) -> Dict[str, Graph]:
        """Fetch all configured ontologies."""
        return {
            prefix: self.get_ontology(prefix) for prefix in self.ontologies if self.get_ontology(prefix) is not None
        }
