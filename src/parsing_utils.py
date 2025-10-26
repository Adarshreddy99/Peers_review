# In: src/parsing_utils.py

import ast
import javalang
import fitz  # PyMuPDF
from typing import List

# --- AST Normalization (Simple, No Compilers) ---

class PythonNormalizer(ast.NodeTransformer):
    """
    Walks the Python AST and replaces all variable and function names
    with generic tokens 'VAR' and 'FUNC'.
    """
    def visit_Name(self, node: ast.Name) -> ast.Name:
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
            node.id = "VAR"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = "FUNC"
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        if isinstance(node.func, ast.Name):
            node.func.id = "FUNC_CALL"
        self.generic_visit(node)
        return node

def normalize_python(code: str) -> str:
    """Parses and normalizes Python code using built-in AST."""
    try:
        tree = ast.parse(code)
        normalizer = PythonNormalizer()
        normalized_tree = normalizer.visit(tree)
        return ast.unparse(normalized_tree)
    except Exception as e:
        print(f"Python AST parsing failed: {e}")
        return ""  # Return empty string on failure

def normalize_java(code: str) -> str:
    """Parses and normalizes Java code using javalang."""
    normalized_tokens = []
    try:
        tokens = javalang.tokenizer.tokenize(code)
        for token in tokens:
            if isinstance(token, javalang.tokenizer.Identifier):
                normalized_tokens.append("ID") # Generic token
            else:
                normalized_tokens.append(token.value)
        return " ".join(normalized_tokens)
    except Exception as e:
        print(f"Java parsing failed: {e}")
        return "" # Return empty string on failure

# --- Document Parsing ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts all text from a PDF given its bytes."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            # Basic cleaning: replace multiple newlines/spaces
            text = ' '.join(text.split())
            return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""

def read_text_file(file_bytes: bytes) -> str:
    """Reads a standard text file, trying common encodings."""
    content_str = ""
    try:
        content_str = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            content_str = file_bytes.decode('latin-1')
        except Exception as e:
            print(f"Text file decoding failed: {e}")
            return ""
    # Basic cleaning
    return ' '.join(content_str.split())