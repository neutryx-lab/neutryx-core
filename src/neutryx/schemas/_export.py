"""JSON Schema and OpenAPI export utilities.

This module provides functionality to export Pydantic schemas to:
- JSON Schema files for validation and documentation
- OpenAPI specification components for API documentation
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode

from neutryx.schemas._registry import SchemaRegistry


class NeutryxJsonSchemaGenerator(GenerateJsonSchema):
    """Custom JSON Schema generator with Neutryx extensions.

    Adds custom metadata fields to generated schemas:
    - $id: Unique schema identifier URL
    - x-schema-version: Schema version from class attribute
    - x-schema-domain: Domain category
    """

    def generate(
        self, schema: Any, mode: JsonSchemaMode = "validation"
    ) -> Dict[str, Any]:
        """Generate JSON Schema with Neutryx extensions."""
        json_schema = super().generate(schema, mode)

        # Add custom metadata if available
        if hasattr(schema, "__name__"):
            json_schema["$id"] = (
                f"https://schemas.neutryx.tech/{schema.__name__}.json"
            )

        return json_schema


def export_json_schema(
    schema_cls: Type[BaseModel],
    output_path: Optional[Path] = None,
    mode: JsonSchemaMode = "serialization",
) -> Dict[str, Any]:
    """Export a single Pydantic model to JSON Schema.

    Args:
        schema_cls: The Pydantic model class
        output_path: Optional file path to write the schema
        mode: 'validation' or 'serialization' mode

    Returns:
        The generated JSON Schema dictionary
    """
    json_schema = schema_cls.model_json_schema(
        mode=mode,
        schema_generator=NeutryxJsonSchemaGenerator,
    )

    # Add Neutryx metadata
    json_schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    json_schema["$id"] = f"https://schemas.neutryx.tech/{schema_cls.__name__}.json"

    if hasattr(schema_cls, "__schema_version__"):
        json_schema["x-schema-version"] = schema_cls.__schema_version__

    if hasattr(schema_cls, "__schema_domain__"):
        json_schema["x-schema-domain"] = schema_cls.__schema_domain__

    json_schema["x-generated-at"] = datetime.utcnow().isoformat()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(json_schema, f, indent=2, ensure_ascii=False)

    return json_schema


def export_json_schemas(
    output_dir: Path,
    schemas: Optional[List[Type[BaseModel]]] = None,
    mode: JsonSchemaMode = "serialization",
    domain: Optional[str] = None,
) -> Dict[str, Path]:
    """Export multiple Pydantic models to JSON Schema files.

    Args:
        output_dir: Directory to write schema files
        schemas: Optional list of schemas; if None, exports all registered
        mode: 'validation' or 'serialization' mode
        domain: Optional domain filter

    Returns:
        Dictionary mapping schema names to output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = SchemaRegistry.instance()

    if schemas is None:
        if domain:
            schemas = registry.by_domain(domain)
        else:
            schemas = registry.all_schemas()

    exported = {}
    for schema_cls in schemas:
        filename = f"{schema_cls.__name__}.json"
        filepath = output_dir / filename

        export_json_schema(schema_cls, filepath, mode)
        exported[schema_cls.__name__] = filepath

    # Write index file
    index = {
        "schemas": list(exported.keys()),
        "generated_at": datetime.utcnow().isoformat(),
        "count": len(exported),
    }
    index_path = output_dir / "_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return exported


def generate_openapi_components(
    schemas: Optional[List[Type[BaseModel]]] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate OpenAPI components section from Pydantic models.

    Generates the 'components/schemas' section of an OpenAPI spec
    that can be merged with FastAPI's generated spec.

    Args:
        schemas: Optional list of schemas; if None, exports all registered
        domain: Optional domain filter

    Returns:
        OpenAPI components dictionary
    """
    registry = SchemaRegistry.instance()

    if schemas is None:
        if domain:
            schemas = registry.by_domain(domain)
        else:
            schemas = registry.all_schemas()

    components: Dict[str, Any] = {"schemas": {}}

    for schema_cls in schemas:
        # Use Pydantic's built-in OpenAPI schema generation
        json_schema = schema_cls.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )

        # Extract nested definitions
        if "$defs" in json_schema:
            for name, definition in json_schema["$defs"].items():
                if name not in components["schemas"]:
                    components["schemas"][name] = definition

        # Add main schema without $defs
        main_schema = {k: v for k, v in json_schema.items() if k != "$defs"}
        components["schemas"][schema_cls.__name__] = main_schema

    return components


def export_openapi_components(
    output_path: Path,
    schemas: Optional[List[Type[BaseModel]]] = None,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Export OpenAPI components to a YAML or JSON file.

    Args:
        output_path: Path to write the file (.json or .yaml)
        schemas: Optional list of schemas
        domain: Optional domain filter

    Returns:
        The generated components dictionary
    """
    components = generate_openapi_components(schemas, domain)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            with output_path.open("w", encoding="utf-8") as f:
                yaml.dump(components, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            # Fall back to JSON if PyYAML not installed
            output_path = output_path.with_suffix(".json")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(components, f, indent=2)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(components, f, indent=2)

    return components


def generate_schema_documentation(
    schema_cls: Type[BaseModel],
) -> str:
    """Generate Markdown documentation for a schema.

    Args:
        schema_cls: The Pydantic model class

    Returns:
        Markdown-formatted documentation string
    """
    lines = [
        f"# {schema_cls.__name__}",
        "",
    ]

    # Docstring
    if schema_cls.__doc__:
        lines.append(schema_cls.__doc__.strip())
        lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Version:** {getattr(schema_cls, '__schema_version__', 'N/A')}")
    lines.append(f"- **Domain:** {getattr(schema_cls, '__schema_domain__', 'N/A')}")
    lines.append(f"- **Module:** {schema_cls.__module__}")
    lines.append("")

    # Fields table
    lines.append("## Fields")
    lines.append("")
    lines.append("| Field | Type | Required | Description |")
    lines.append("|-------|------|----------|-------------|")

    for name, field in schema_cls.model_fields.items():
        type_str = str(field.annotation).replace("<", "&lt;").replace(">", "&gt;")
        # Clean up type string
        type_str = type_str.replace("typing.", "").replace("neutryx.schemas.", "")
        required = "Yes" if field.is_required() else "No"
        desc = (field.description or "").replace("|", "\\|")
        lines.append(f"| `{name}` | `{type_str}` | {required} | {desc} |")

    lines.append("")

    return "\n".join(lines)
