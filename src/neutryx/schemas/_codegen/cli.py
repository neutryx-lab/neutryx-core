"""CLI for schema code generation.

Usage:
    python -m neutryx.schemas._codegen.cli json -o ./schemas/
    python -m neutryx.schemas._codegen.cli typescript -o ./types.ts
    python -m neutryx.schemas._codegen.cli openapi -o ./openapi.yaml
    python -m neutryx.schemas._codegen.cli docs -o ./docs/schemas/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Type

from pydantic import BaseModel


def get_registry():
    """Lazy import of registry to avoid circular imports."""
    from neutryx.schemas._registry import SchemaRegistry

    return SchemaRegistry.instance()


def get_export_functions():
    """Lazy import of export functions."""
    from neutryx.schemas._export import (
        export_json_schemas,
        export_openapi_components,
        generate_schema_documentation,
    )

    return export_json_schemas, export_openapi_components, generate_schema_documentation


def generate_typescript_types(
    schemas: List[Type[BaseModel]],
    output_file: Path,
    namespace: str = "Neutryx",
) -> None:
    """Generate TypeScript interfaces from Pydantic models.

    Args:
        schemas: List of Pydantic model classes
        output_file: Path to write TypeScript file
        namespace: TypeScript namespace to wrap types in
    """
    lines = [
        "// Auto-generated TypeScript types from Neutryx schemas",
        f"// Generated at: {datetime.utcnow().isoformat()}",
        "// DO NOT EDIT MANUALLY - regenerate with: python -m neutryx.schemas._codegen.cli typescript",
        "",
        f"export namespace {namespace} {{",
        "",
    ]

    for schema_cls in sorted(schemas, key=lambda s: s.__name__):
        json_schema = schema_cls.model_json_schema()
        ts_lines = _json_schema_to_typescript(schema_cls.__name__, json_schema)
        lines.extend(ts_lines)

    lines.append("}")
    lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


def _json_schema_to_typescript(name: str, schema: Dict[str, Any]) -> List[str]:
    """Convert JSON Schema to TypeScript interface."""
    lines = []

    # Add description as JSDoc if available
    if "description" in schema:
        lines.append(f"  /** {schema['description']} */")

    lines.append(f"  export interface {name} {{")

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    for prop_name, prop_schema in properties.items():
        ts_type = _json_type_to_ts(prop_schema)
        optional = "" if prop_name in required else "?"

        # Add field description
        if "description" in prop_schema:
            lines.append(f"    /** {prop_schema['description']} */")

        lines.append(f"    {prop_name}{optional}: {ts_type};")

    lines.append("  }")
    lines.append("")

    return lines


def _json_type_to_ts(schema: Dict[str, Any]) -> str:
    """Map JSON Schema type to TypeScript type."""
    # Handle references
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return ref

    # Handle anyOf (Optional types)
    if "anyOf" in schema:
        types = []
        for option in schema["anyOf"]:
            if option.get("type") == "null":
                continue
            types.append(_json_type_to_ts(option))
        if len(types) == 1:
            return f"{types[0]} | null"
        return " | ".join(types) + " | null"

    # Handle enums
    if "enum" in schema:
        return " | ".join(f'"{v}"' for v in schema["enum"])

    # Handle arrays
    if schema.get("type") == "array":
        items = schema.get("items", {})
        item_type = _json_type_to_ts(items)
        return f"Array<{item_type}>"

    # Handle objects
    if schema.get("type") == "object":
        additional = schema.get("additionalProperties")
        if additional:
            value_type = _json_type_to_ts(additional)
            return f"Record<string, {value_type}>"
        return "Record<string, unknown>"

    # Basic type mapping
    type_mapping = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "null": "null",
    }

    json_type = schema.get("type", "unknown")

    # Handle format hints
    if json_type == "string":
        fmt = schema.get("format")
        if fmt in ("date", "date-time"):
            return "string"  # ISO date strings
        if fmt == "uuid":
            return "string"

    return type_mapping.get(json_type, "unknown")


def cmd_json(args: argparse.Namespace) -> int:
    """Export JSON Schemas."""
    export_json_schemas, _, _ = get_export_functions()
    registry = get_registry()

    schemas = None
    if args.domain:
        schemas = registry.by_domain(args.domain)
        if not schemas:
            print(f"No schemas found in domain '{args.domain}'", file=sys.stderr)
            return 1

    exported = export_json_schemas(args.output, schemas, domain=args.domain)
    print(f"Exported {len(exported)} JSON schemas to {args.output}")
    return 0


def cmd_typescript(args: argparse.Namespace) -> int:
    """Generate TypeScript types."""
    registry = get_registry()

    schemas = registry.all_schemas()
    if args.domain:
        schemas = registry.by_domain(args.domain)

    if not schemas:
        print("No schemas found", file=sys.stderr)
        return 1

    generate_typescript_types(schemas, args.output, args.namespace)
    print(f"Generated TypeScript types for {len(schemas)} schemas to {args.output}")
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    """Export OpenAPI components."""
    _, export_openapi_components, _ = get_export_functions()
    registry = get_registry()

    schemas = None
    if args.domain:
        schemas = registry.by_domain(args.domain)

    components = export_openapi_components(args.output, schemas, domain=args.domain)
    schema_count = len(components.get("schemas", {}))
    print(f"Exported OpenAPI components with {schema_count} schemas to {args.output}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    """Generate documentation."""
    _, _, generate_schema_documentation = get_export_functions()
    registry = get_registry()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    schemas = registry.all_schemas()
    if args.domain:
        schemas = registry.by_domain(args.domain)

    for schema_cls in schemas:
        doc = generate_schema_documentation(schema_cls)
        filepath = output_dir / f"{schema_cls.__name__}.md"
        filepath.write_text(doc, encoding="utf-8")

    print(f"Generated documentation for {len(schemas)} schemas to {output_dir}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List registered schemas."""
    registry = get_registry()

    if args.domain:
        schemas = registry.by_domain(args.domain)
        print(f"Schemas in domain '{args.domain}':")
    else:
        schemas = registry.all_schemas()
        print("All registered schemas:")

    for schema in sorted(schemas, key=lambda s: s.__name__):
        domain = getattr(schema, "__schema_domain__", "unknown")
        version = getattr(schema, "__schema_version__", "?")
        print(f"  {schema.__name__:30} [{domain}] v{version}")

    print(f"\nTotal: {len(schemas)} schemas")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Neutryx Schema Code Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s json -o ./generated/json_schemas/
  %(prog)s typescript -o ./generated/types.ts
  %(prog)s openapi -o ./generated/openapi.yaml
  %(prog)s docs -o ./docs/schemas/
  %(prog)s list
  %(prog)s list --domain trading
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # JSON Schema export
    json_parser = subparsers.add_parser("json", help="Export JSON Schemas")
    json_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output directory"
    )
    json_parser.add_argument("--domain", type=str, help="Filter by domain")
    json_parser.set_defaults(func=cmd_json)

    # TypeScript generation
    ts_parser = subparsers.add_parser("typescript", help="Generate TypeScript types")
    ts_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output file"
    )
    ts_parser.add_argument(
        "--namespace", default="Neutryx", help="TypeScript namespace"
    )
    ts_parser.add_argument("--domain", type=str, help="Filter by domain")
    ts_parser.set_defaults(func=cmd_typescript)

    # OpenAPI export
    openapi_parser = subparsers.add_parser(
        "openapi", help="Export OpenAPI components"
    )
    openapi_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output file (.json or .yaml)"
    )
    openapi_parser.add_argument("--domain", type=str, help="Filter by domain")
    openapi_parser.set_defaults(func=cmd_openapi)

    # Documentation generation
    docs_parser = subparsers.add_parser("docs", help="Generate schema documentation")
    docs_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output directory"
    )
    docs_parser.add_argument("--domain", type=str, help="Filter by domain")
    docs_parser.set_defaults(func=cmd_docs)

    # List schemas
    list_parser = subparsers.add_parser("list", help="List registered schemas")
    list_parser.add_argument("--domain", type=str, help="Filter by domain")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
