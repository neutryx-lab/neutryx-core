"""
audit_sdd_gap.py

A utility script to perform a gap analysis between the implementation codebase
(Python/JAX) and the Security Design Description (SDD) document.

This tool utilises AST parsing to extract structural definitions from the source
code and compares them against the entities described in the Markdown-formatted SDD.

Usage:
    python audit_sdd_gap.py --code ./src --sdd ./docs/SDD.md --output gap_report.json
"""

import ast
import os
import re
import json
import argparse
from typing import Set, Dict, List, Tuple
from pathlib import Path

class CodeBaseAnalyser(ast.NodeVisitor):
    """
    Parses Python source files to extract class and function definitions.
    """
    def __init__(self, ignore_private: bool = True):
        self.definitions: Set[str] = set()
        self.ignore_private = ignore_private
        self.current_file = ""

    def visit_ClassDef(self, node: ast.ClassDef):
        if self.ignore_private and node.name.startswith("_"):
            return
        self.definitions.add(node.name)
        # Continue traversing to find methods within the class
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.ignore_private and node.name.startswith("_"):
            return
        # In a flat functional design (common in JAX), top-level functions are key.
        self.definitions.add(node.name)

    def analyse_directory(self, root_path: str) -> Set[str]:
        """
        Walks through the directory recursively and parses all .py files.
        Returns a set of all found identifiers (classes and functions).
        """
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    self.current_file = full_path
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            tree = ast.parse(f.read(), filename=full_path)
                            self.visit(tree)
                    except SyntaxError:
                        print(f"Warning: Syntax error in file {full_path}. Skipped.")
        return self.definitions


class SDDParser:
    """
    Parses Markdown files within a directory (or a single file) to extract 
    mentioned modules and components.
    Capable of scanning recursively through .kiro folder structures.
    """
    def __init__(self, sdd_path: str):
        self.sdd_path = sdd_path
        self.mentioned_entities: Set[str] = set()

    def parse_content(self, content: str):
        """Extract entities from a single string of content."""
        # Strategy 1: Extract headers (e.g., ## Authenticator)
        headers = re.findall(r'^#+\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
        
        # Strategy 2: Extract code blocks or backticked items
        code_snippets = re.findall(r'`([a-zA-Z0-9_]+)`', content)

        # Strategy 3: Extract Python-like function signatures
        signatures = re.findall(r'def\s+([a-zA-Z0-9_]+)', content)

        self.mentioned_entities.update(headers)
        self.mentioned_entities.update(code_snippets)
        self.mentioned_entities.update(signatures)

    def parse(self) -> Set[str]:
        path_obj = Path(self.sdd_path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {self.sdd_path}")

        # Case 1: Directory (.kiro folder)
        if path_obj.is_dir():
            print(f"Scanning directory: {self.sdd_path} ...")
            for root, _, files in os.walk(self.sdd_path):
                for file in files:
                    # Parse only markdown or text files, ignore binary/json for now unless needed
                    if file.endswith((".md", ".txt", ".markdown")):
                        full_path = os.path.join(root, file)
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                self.parse_content(f.read())
                        except Exception as e:
                            print(f"Warning: Could not read {full_path}: {e}")
        
        # Case 2: Single file
        else:
            with open(path_obj, "r", encoding="utf-8") as f:
                self.parse_content(f.read())

        return self.mentioned_entities


class GapReporter:
    """
    Compares the codebase and the SDD to generate a discrepancy report.
    """
    def __init__(self, code_entities: Set[str], sdd_entities: Set[str]):
        self.code_entities = code_entities
        self.sdd_entities = sdd_entities

    def generate_report(self) -> Dict[str, List[str]]:
        # Items in Code but NOT in SDD (Risk: Undocumented functionality)
        undocumented = sorted(list(self.code_entities - self.sdd_entities))
        
        # Items in SDD but NOT in Code (Risk: Vapourware or incorrect naming)
        unimplemented = sorted(list(self.sdd_entities - self.code_entities))

        return {
            "summary": {
                "total_code_entities": len(self.code_entities),
                "total_sdd_entities": len(self.sdd_entities),
                "undocumented_count": len(undocumented),
                "unimplemented_count": len(unimplemented)
            },
            "undocumented_items_in_code": undocumented,
            "unimplemented_items_in_sdd": unimplemented
        }


def main():
    parser = argparse.ArgumentParser(description="Audit SDD against Codebase")
    parser.add_argument("--code", required=True, help="Path to the source code directory")
    parser.add_argument("--sdd", required=True, help="Path to the SDD markdown file")
    parser.add_argument("--output", default="gap_report.json", help="Output JSON file path")
    args = parser.parse_args()

    print(f"Analysing codebase at: {args.code}")
    analyser = CodeBaseAnalyser(ignore_private=True)
    code_entities = analyser.analyse_directory(args.code)

    print(f"Parsing SDD at: {args.sdd}")
    sdd_parser = SDDParser(args.sdd)
    sdd_entities = sdd_parser.parse()

    print("Comparing definitions...")
    reporter = GapReporter(code_entities, sdd_entities)
    report = reporter.generate_report()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Analysis complete. Report saved to {args.output}")
    print(f"Undocumented items found: {report['summary']['undocumented_count']}")
    print(f"Unimplemented items found: {report['summary']['unimplemented_count']}")

if __name__ == "__main__":
    main()
    