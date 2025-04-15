import logging
import datetime
import urllib.parse
import ifcopenshell
from pathlib import Path
from pyshacl import validate
from typing import Dict, Optional
from .common_converter import OntologyManager
from rdflib.namespace import RDF, RDFS, XSD, SH
from rdflib import Graph, Namespace, Literal, BNode


def encode_uri_component(s: str) -> str:
    """Safely encode URI component, preserving IFC GlobalId characters."""
    return urllib.parse.quote(s, safe="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_")


def get_typed_literal(value):
    """Convert value to appropriate XSD type."""
    try:
        float_val = float(value)
        # Check if it's actually an integer
        if float_val.is_integer():
            return Literal(int(float_val), datatype=XSD.integer)
        return Literal(float_val, datatype=XSD.decimal)
    except (ValueError, TypeError):
        # Check for boolean values
        if value.lower() in ("true", "false"):
            return Literal(value.lower() == "true", datatype=XSD.boolean)
        return Literal(str(value), datatype=XSD.string)


def get_project_units(ifc) -> Dict[str, str]:
    """Extract unit information from IFC project."""
    units = {}
    project = ifc.by_type("IfcProject")[0]
    for context in project.UnitsInContext.Units:
        if context.is_a("IfcNamedUnit"):
            unit_type = context.UnitType
            if context.is_a("IfcSIUnit"):
                unit_name = context.Name
                if context.Prefix:
                    unit_name = f"{context.Prefix}{unit_name}"
                units[unit_type] = unit_name
    return units


def get_property_unit_type(prop) -> Optional[str]:
    """Get the unit type for a property."""
    if hasattr(prop, "Unit") and prop.Unit:
        return prop.Unit.UnitType

    # Enhanced unit type inference
    area_keywords = {"area", "surface"}
    volume_keywords = {"volume", "capacity"}
    length_keywords = {"length", "width", "height", "depth", "offset", "perimeter"}
    mass_keywords = {"mass", "weight"}
    temperature_keywords = {"temperature", "thermal"}

    prop_name = prop.Name.lower()
    if any(keyword in prop_name for keyword in area_keywords):
        return "AREAUNIT"
    elif any(keyword in prop_name for keyword in volume_keywords):
        return "VOLUMEUNIT"
    elif any(keyword in prop_name for keyword in length_keywords):
        return "LENGTHUNIT"
    elif any(keyword in prop_name for keyword in mass_keywords):
        return "MASSUNIT"
    elif any(keyword in prop_name for keyword in temperature_keywords):
        return "THERMODYNAMICTEMPERATUREUNIT"
    return None


def properties_from_ifc(ifc):
    project_units = get_project_units(ifc)

    for relation in ifc.by_type("IfcRelDefinesByProperties"):
        property_set = relation.RelatingPropertyDefinition
        pset_name = property_set.Name
        objects = relation.RelatedObjects

        if property_set.is_a("IfcPropertySet"):
            for property in property_set.HasProperties:
                property_name = property.Name
                unit_type = get_property_unit_type(property)
                unit = project_units.get(unit_type) if unit_type else None

                for attr in (x for x in dir(property) if x.endswith(("Value", "Values"))):
                    values = getattr(property, attr)
                    if isinstance(values, str) or not hasattr(values, "__iter__"):
                        values = [values]
                    for value in values:
                        if value is not None:
                            if hasattr(value, "wrappedValue"):
                                value = value.wrappedValue
                            for object in objects:
                                yield (object.GlobalId, pset_name, property_name, value, unit, unit_type)


def get_ifc_schema_version(ifc) -> str:
    """Get the schema version of the IFC file."""
    return ifc.schema


def get_containment_structure(space, schema_version: str):
    """Get containment structure based on IFC schema version."""
    if schema_version == "IFC2X3":
        # Try different possible attribute names for IFC2X3
        if hasattr(space, "Decomposes"):
            # This is the correct inverse relationship name in IFC2X3
            return [rel for rel in space.Decomposes if rel.is_a("IfcRelContainedInSpatialStructure")]
        return []
    else:
        # IFC4 and later have direct ContainedInStructure attribute
        if hasattr(space, "ContainedInStructure"):
            return space.ContainedInStructure
        return []


class ValidationError(Exception):
    """Exception raised when validation fails."""

    pass


class IfcConverter:
    """Converts IFC files to TTL format with BOT ontology integration."""

    # Define both URLs and formats for each ontology
    ONTOLOGIES = {
        "ifc": ("http://ifcowl.openbimstandards.org/IFC2X3#", "xml"),  # IFC typically uses RDF/XML
        "inst": ("http://example.org/instance#", "turtle"),  # Local namespace
        "prop": ("http://example.org/property#", "turtle"),  # Local namespace
        "pset": ("http://ifcowl.openbimstandards.org/IFC2X3/PropertySet#", "xml"),
        "bot": ("https://w3id.org/bot#", "turtle"),  # BOT uses turtle
        "unit": ("http://ifcowl.openbimstandards.org/IFC2X3/UNIT#", "xml"),
        "qudt": ("http://qudt.org/schema/qudt/", "turtle"),
        "qudt-unit": ("http://qudt.org/vocab/unit/", "turtle"),
        "qudt-qk": ("http://qudt.org/vocab/quantitykind/", "turtle"),
        "prov": ("http://www.w3.org/ns/prov#", "turtle"),
        "dcterms": ("http://purl.org/dc/terms/", "turtle"),
    }

    # Keep NAMESPACES for easy access to URLs
    NAMESPACES = {prefix: url for prefix, (url, _) in ONTOLOGIES.items()}

    UNIT_MAPPINGS = {
        ("LENGTHUNIT", "METRE"): ("M", "Length"),
        ("AREAUNIT", "SQUARE_METRE"): ("M2", "Area"),
        ("VOLUMEUNIT", "CUBIC_METRE"): ("M3", "Volume"),
        ("MASSUNIT", "GRAM"): ("G", "Mass"),
        ("MASSUNIT", "KILOGRAM"): ("KG", "Mass"),
        ("TIMEUNIT", "SECOND"): ("SEC", "Time"),
        ("THERMODYNAMICTEMPERATUREUNIT", "KELVIN"): ("K", "Temperature"),
        ("THERMODYNAMICTEMPERATUREUNIT", "CELSIUS"): ("DEG_C", "Temperature"),
        ("PRESSUREUNIT", "PASCAL"): ("PA", "Pressure"),
    }

    UNIT_KEYWORDS = {
        "AREAUNIT": {"area", "surface"},
        "VOLUMEUNIT": {"volume", "capacity"},
        "LENGTHUNIT": {"length", "width", "height", "depth", "offset", "perimeter"},
        "MASSUNIT": {"mass", "weight"},
        "THERMODYNAMICTEMPERATUREUNIT": {"temperature", "thermal"},
    }

    BOT_MAPPINGS = {
        "IfcBuilding": "Building",
        "IfcBuildingStorey": "Storey",
        "IfcSpace": "Space",
    }

    BOT_ELEMENTS = {"IfcWall", "IfcSlab", "IfcRoof", "IfcBeam", "IfcColumn"}
    BOT_INTERFACES = {"IfcWindow", "IfcDoor"}

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the IFC converter."""
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data"

        # Initialize ontology manager with proper ontology definitions
        self.ontology_manager = OntologyManager(cache_dir=str(self.data_dir / "ontologies"), ontologies=self.ONTOLOGIES)

        # Initialize namespaces using the NAMESPACES dict
        self.ns = {prefix: Namespace(url) for prefix, url in self.NAMESPACES.items()}
        self.project_units = {}

    def _get_unit_type(self, name: str) -> Optional[str]:
        """Infer unit type from property name."""
        name = name.lower()
        return next((unit for unit, kw in self.UNIT_KEYWORDS.items() if any(k in name for k in kw)), None)

    def _add_property(self, g: Graph, subject, name: str, value, unit_type: str = None, unit_name: str = None):
        """Add a property with optional unit information."""
        node = BNode()
        safe_name = name.replace(" ", "_")
        g.add((subject, self.ns["prop"][safe_name], node))

        # Add value with appropriate type
        try:
            float_val = float(value)
            lit_value = Literal(
                int(float_val) if float_val.is_integer() else float_val,
                datatype=XSD.integer if float_val.is_integer() else XSD.decimal,
            )
        except (ValueError, TypeError):
            is_bool = str(value).lower() in ("true", "false")
            lit_value = Literal(
                str(value).lower() == "true" if is_bool else str(value), datatype=XSD.boolean if is_bool else XSD.string
            )
        g.add((node, RDF.value, lit_value))

        # Add unit information if available
        if unit_type and unit_name:
            g.add((node, self.ns["unit"].unitType, Literal(unit_type)))
            g.add((node, self.ns["unit"].unitName, Literal(unit_name)))
            if (unit_type, unit_name) in self.UNIT_MAPPINGS:
                qudt_unit, qudt_qk = self.UNIT_MAPPINGS[(unit_type, unit_name)]
                g.add((node, self.ns["qudt"].unit, self.ns["qudt-unit"][qudt_unit]))
                g.add((node, self.ns["qudt"].quantityKind, self.ns["qudt-qk"][qudt_qk]))

    def validate_graph(self, g: Graph) -> None:
        """Validate the generated RDF graph using SHACL shapes."""
        # Load SHACL shapes
        shapes_graph = Graph()
        shapes_file = Path(__file__).parent / "ifc_converter_shapes.ttl"
        shapes_graph.parse(shapes_file, format="turtle")

        try:
            # Perform SHACL validation with minimal settings
            conforms, results_graph, results_text = validate(
                data_graph=g,
                shacl_graph=shapes_graph,
                ont_graph=None,  # Disable ontology graph
                inference=False,  # Disable inference
                debug=False,  # Disable debug output
                serialize_report_graph=False,  # Keep results as Graph object
            )

            if not conforms:
                # Filter and format validation results
                validation_errors = []
                for result in results_graph.subjects(RDF.type, SH.ValidationResult):
                    severity = results_graph.value(result, SH.resultSeverity)

                    # Only treat Violations as errors, Warnings can be ignored
                    if severity == SH.Violation:
                        focus_node = results_graph.value(result, SH.focusNode)
                        path = results_graph.value(result, SH.resultPath)
                        message = results_graph.value(result, SH.resultMessage)

                        error_msg = f"Node {focus_node}"
                        if path:
                            error_msg += f" - Path: {path}"
                        if message:
                            error_msg += f" - Message: {message}"

                        validation_errors.append(error_msg)

                if validation_errors:
                    raise ValidationError("Graph validation failed:\n" + "\n".join(validation_errors))
                else:
                    print("Validation completed with warnings only")

        except Exception as e:
            if "Invalid codepoint in stream" in str(e):
                print("Warning: Found invalid characters in literals. Attempting to clean data...")
                self._clean_literal_values(g)
                return self.validate_graph(g)
            else:
                raise ValidationError(f"Validation error: {str(e)}")

    def _clean_literal_values(self, g: Graph):
        """Clean literal values in the graph to remove invalid characters."""
        for s, p, o in g:
            if isinstance(o, Literal):
                # Get the literal value
                value = str(o)
                # Clean the value by removing or replacing invalid characters
                cleaned_value = "".join(char for char in value if ord(char) < 0x10000)
                if value != cleaned_value:
                    # Remove the old triple
                    g.remove((s, p, o))
                    # Add the new triple with cleaned value
                    g.add((s, p, Literal(cleaned_value, datatype=o.datatype, lang=o.language)))

    def convert(self, ifc_filename: str, ttl_filename: str) -> None:
        """Convert IFC file to TTL format with validation."""
        print(f"Converting {ifc_filename} to {ttl_filename}")  # Add progress indication

        ifc = ifcopenshell.open(ifc_filename)
        schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name(ifc.schema)

        # Store the current schema version
        self.current_schema = ifc.schema
        print(f"Using IFC schema: {self.current_schema}")  # Add schema info

        g = Graph()
        for prefix, ns in self.ns.items():
            g.bind(prefix, ns)

        # Get project units
        project = ifc.by_type("IfcProject")[0]
        self.project_units = {
            unit.UnitType: unit.Name for unit in project.UnitsInContext.Units if unit.is_a("IfcSIUnit")
        }

        # Add provenance
        prov_node = BNode()
        g.add((prov_node, RDF.type, self.ns["prov"].Activity))
        g.add(
            (
                prov_node,
                self.ns["prov"].startedAtTime,
                Literal(datetime.datetime.now().isoformat(), datatype=XSD.dateTime),
            )
        )
        g.add((prov_node, self.ns["prov"].used, Literal(f"ifcopenshell {ifcopenshell.version}")))
        g.add((prov_node, self.ns["dcterms"].source, Literal(str(ifc_filename))))

        # Process elements
        for element in ifc.by_type("IfcProduct"):
            subject = self.ns["inst"][urllib.parse.quote(element.GlobalId)]
            elem_type = element.is_a()

            # Get the entity from schema
            entity = schema.declaration_by_name(elem_type)

            # Add the element type as an ENTITY
            class_uri = self.ns["ifc"][elem_type]
            g.add((class_uri, RDF.type, self.ns["ifc"].ENTITY))

            # Add the element as an instance
            g.add((subject, RDF.type, class_uri))

            # Add supertype information
            if entity.supertype():
                supertype_uri = self.ns["ifc"][entity.supertype().name()]
                g.add((class_uri, RDFS.subClassOf, supertype_uri))
                g.add((supertype_uri, RDF.type, self.ns["ifc"].ENTITY))

            # Add label
            if element.Name:
                g.add((subject, RDFS.label, Literal(element.Name)))

            # Add BOT typing (this remains valid as it's a separate ontology)
            if elem_type in self.BOT_MAPPINGS:
                g.add((subject, RDF.type, self.ns["bot"][self.BOT_MAPPINGS[elem_type]]))
            elif elem_type in self.BOT_ELEMENTS:
                g.add((subject, RDF.type, self.ns["bot"].Element))
            elif elem_type in self.BOT_INTERFACES:
                g.add((subject, RDF.type, self.ns["bot"].Interface))

            # Process properties
            for rel in element.IsDefinedBy:
                if not rel.is_a("IfcRelDefinesByProperties"):
                    continue

                pset = rel.RelatingPropertyDefinition
                if not pset.is_a("IfcPropertySet"):
                    continue

                # Create property set node
                pset_node = BNode()
                g.add((subject, self.ns["ifc"].hasPropertySet, pset_node))

                # Add property set type
                pset_type = self.ns["pset"][urllib.parse.quote(pset.Name.replace(" ", "_"))]
                g.add((pset_type, RDF.type, self.ns["ifc"].IfcPropertySet))
                g.add((pset_node, RDF.type, pset_type))

                # Process properties
                for prop in pset.HasProperties:
                    if not (hasattr(prop, "NominalValue") and prop.NominalValue):
                        continue

                    unit_type = (
                        prop.Unit.UnitType if hasattr(prop, "Unit") and prop.Unit else self._get_unit_type(prop.Name)
                    )
                    unit_name = (
                        prop.Unit.Name if hasattr(prop, "Unit") and prop.Unit else self.project_units.get(unit_type)
                    )

                    # Add unit if available
                    if unit_type:
                        unit_individual = self.ns["ifc"][unit_type]
                        g.add((unit_individual, RDF.type, self.ns["ifc"].IfcUnitEnum))

                    self._add_property(g, pset_node, prop.Name, prop.NominalValue.wrappedValue, unit_type, unit_name)

        # Validate immediately after graph creation
        print("Validating graph...")
        try:
            self.validate_graph(g)
            print("Validation successful")
        except ValidationError as e:
            print(f"Validation failed: {str(e)}")
            raise  # Re-raise the error to stop processing

        # Only reaches here if validation passed
        print(f"Saving validated graph to {ttl_filename}")
        g.serialize(destination=ttl_filename, format="turtle")
        print("Conversion complete")


def ifc_to_ttl(ifc_filename: str, ttl_filename: str):
    """Convert IFC file to TTL format."""
    converter = IfcConverter()
    # Remove try/except to allow errors to propagate up
    converter.convert(ifc_filename, ttl_filename)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    root_dir = Path(__file__).parent.parent.parent / "data"
    project_names = ["buildingsmart_duplex", "buildingsmart_dental", "buildingsmart_schependomlaan"]
    # project_names = ["buildingsmart_schependomlaan"]

    for project_name in project_names:
        raw_dir = root_dir / "raw" / project_name
        graph_dir = root_dir / "graph" / project_name
        graph_dir.mkdir(parents=True, exist_ok=True)

        for ifc_filepath in raw_dir.glob("*.ifc"):
            ttl_filepath = graph_dir / ifc_filepath.name.replace(".ifc", ".ttl")
            print(f"\nProcessing {ifc_filepath.name}")

            # Remove try/except to allow errors to propagate
            ifc_to_ttl(ifc_filepath, ttl_filepath)
