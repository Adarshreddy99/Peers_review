# In: src/parsing_utils.py

import ast
import javalang
import fitz  # PyMuPDF
from typing import List, Dict, Set, Tuple # Added Tuple
import hashlib # Can be used for fingerprinting, but let's stick to features for now

# --- SmartPythonNormalizer Class (Unchanged from your version) ---
class SmartPythonNormalizer(ast.NodeTransformer):
    """
    Smarter normalization that preserves semantic information while
    catching plagiarism. Keeps important structural features.
    """
    def __init__(self):
        self.function_names = set()
        self.class_names = set()
        self.import_modules = set()
        self.control_flow_pattern = []
        self.api_calls = set()
        # Keep track of variable scope if needed (more advanced)
        # self.scope_stack = [{}]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Only normalize likely user-defined variables, keep known APIs/libraries"""
        if isinstance(node.ctx, (ast.Load, ast.Store, ast.Param)):
            common_apis = {
                'np', 'pd', 'torch', 'tf', 'plt', 'os', 'sys', 'math',
                'random', 'time', 'json', 'requests', 'print', 'len',
                'range', 'enumerate', 'zip', 'map', 'filter', 'sum',
                'max', 'min', 'abs', 'int', 'str', 'float', 'list',
                'dict', 'set', 'tuple', 'open', 'file', 'self', 'cls',
                'Exception', 'ValueError', 'TypeError', 'IndexError', # Common exceptions
                '__name__', # Often used in if __name__ == "__main__":
                # Add more standard library or framework names if needed
            }
            # Normalize if it's not a common API/builtin and not already generic
            if node.id not in common_apis and node.id not in ["VAR", "FUNC", "CLASS", "FUNC_CALL"]:
                 node.id = "VAR"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Keep function signature structure, normalize name"""
        self.function_names.add(node.name)
        # Preserve main/special functions
        if node.name not in ['__init__', '__main__', 'main', '__str__', '__repr__']:
            node.name = "FUNC"
        # Record function complexity (number of arguments)
        self.control_flow_pattern.append(f"FUNC_{len(node.args.args)}_args")
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Keep class structure"""
        self.class_names.add(node.name)
        node.name = "CLASS"
        self.control_flow_pattern.append("CLASS_DEF")
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.Call:
        """Track API/function calls but normalize user functions"""
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute): # Handle method calls like obj.method()
             # Try to capture the method name
             # This can get complex with nested attributes
             if isinstance(node.func.value, ast.Name):
                 # Simple obj.method
                 func_name = f"{node.func.value.id}.{node.func.attr}"
             else:
                 # More complex like obj.sub.method - just capture method name
                 func_name = node.func.attr
        
        if func_name:
             self.api_calls.add(func_name)
             # Don't normalize built-in/common API calls aggressively
             # Check only the final part (method name) against common patterns
             method_part = func_name.split('.')[-1]
             common_calls = {'print', 'len', 'range', 'open', 'enumerate', 'zip', 'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'copy', 'sort', 'reverse', 'get', 'keys', 'values', 'items', 'update', 'add', 'read', 'write', 'close', 'format', 'join', 'split', 'strip', 'replace'}
             # If it's a Name node (simple function call) and not common, normalize it
             if isinstance(node.func, ast.Name) and func_name not in common_calls:
                 node.func.id = "FUNC_CALL"
             # If it's an Attribute node (method call), maybe normalize the object part if it's a variable?
             # Let's keep method calls less normalized for now to preserve structure
             # if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
             #     self.visit(node.func.value) # Normalize the object variable if needed

        self.generic_visit(node)
        return node

    def visit_Import(self, node: ast.Import) -> ast.Import:
        """Track imported modules"""
        for alias in node.names:
            self.import_modules.add(alias.name)
        return node # Keep imports in the normalized code

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        """Track from imports"""
        if node.module:
            self.import_modules.add(node.module)
        # Maybe track specific imported names too? e.g., from math import sqrt
        # for alias in node.names:
        #     self.api_calls.add(alias.name)
        return node # Keep imports

    # --- Control Flow Tracking (Unchanged) ---
    def visit_If(self, node: ast.If) -> ast.If:
        self.control_flow_pattern.append("IF")
        self.generic_visit(node)
        return node
    def visit_For(self, node: ast.For) -> ast.For:
        self.control_flow_pattern.append("FOR")
        self.generic_visit(node)
        return node
    def visit_While(self, node: ast.While) -> ast.While:
        self.control_flow_pattern.append("WHILE")
        self.generic_visit(node)
        return node
    def visit_Try(self, node: ast.Try) -> ast.Try:
        self.control_flow_pattern.append("TRY")
        self.generic_visit(node)
        return node
    # --- End Control Flow ---

# --- Feature Extraction Function (Unchanged from your version) ---
def extract_structural_features(code: str) -> Dict[str, any]:
    """
    Extract high-level structural features that differentiate projects.
    Returns a feature dictionary for comparison.
    """
    try:
        tree = ast.parse(code)
    except Exception as e:
        print(f"Feature extraction parse failed: {e}")
        return {} # Return empty if code can't be parsed

    features = {
        'num_functions': 0,
        'num_classes': 0,
        'num_imports': 0,
        'num_loops': 0,
        'num_conditionals': 0,
        'num_try_except': 0,
        'max_nesting_depth': 0,
        'avg_function_length': 0.0, # Use float
        'import_categories': set(),
        'control_flow_complexity': 0 # Simple count of branches/loops
    }

    class FeatureExtractor(ast.NodeVisitor):
        def __init__(self):
            self.current_depth = 0
            self.max_depth = 0
            self.function_lengths = []
            # Track depth within blocks, not just function/class defs
            self.block_starters = (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.Try, ast.With)

        def generic_visit(self, node):
            """Track nesting depth"""
            is_block = isinstance(node, self.block_starters)
            if is_block:
                self.current_depth += 1
                self.max_depth = max(self.max_depth, self.current_depth)

            super().generic_visit(node) # Visit children first

            if is_block:
                self.current_depth -= 1


        def visit_FunctionDef(self, node):
            features['num_functions'] += 1
            # Calculate length more accurately (lines of code, excluding comments/blanks if possible)
            # Simple approximation: number of direct statements in the body
            func_len = len(node.body)
            self.function_lengths.append(func_len)
            self.generic_visit(node) # Call generic first to handle nesting

        def visit_ClassDef(self, node):
            features['num_classes'] += 1
            self.generic_visit(node)

        def visit_Import(self, node):
            features['num_imports'] += len(node.names) # Count each imported item
            for alias in node.names:
                 # Categorize imports (simplified)
                 if any(lib in alias.name for lib in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn']):
                     features['import_categories'].add('data_science')
                 elif any(lib in alias.name for lib in ['torch', 'tensorflow', 'keras']):
                     features['import_categories'].add('deep_learning')
                 elif any(lib in alias.name for lib in ['flask', 'django', 'requests', 'fastapi', 'streamlit', 'gradio']):
                     features['import_categories'].add('web')
                 elif alias.name in ['os', 'sys', 'math', 'json', 're', 'datetime', 'collections']:
                      features['import_categories'].add('stdlib_common')
                 else:
                      features['import_categories'].add('other')

            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            features['num_imports'] += len(node.names) # Count specific imports
            if node.module:
                 if any(lib in node.module for lib in ['numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn']):
                     features['import_categories'].add('data_science')
                 elif any(lib in node.module for lib in ['torch', 'tensorflow', 'keras']):
                     features['import_categories'].add('deep_learning')
                 elif any(lib in node.module for lib in ['flask', 'django', 'requests', 'fastapi', 'streamlit', 'gradio']):
                     features['import_categories'].add('web')
                 elif node.module in ['os', 'sys', 'math', 'json', 're', 'datetime', 'collections']:
                      features['import_categories'].add('stdlib_common')
                 else:
                      features['import_categories'].add('other')
            self.generic_visit(node)

        def visit_For(self, node):
            features['num_loops'] += 1
            features['control_flow_complexity'] += 1
            self.generic_visit(node)

        def visit_While(self, node):
            features['num_loops'] += 1
            features['control_flow_complexity'] += 1
            self.generic_visit(node)

        def visit_If(self, node):
            features['num_conditionals'] += 1
            features['control_flow_complexity'] += 1
            # Add complexity for elif/else branches
            if node.orelse:
                features['control_flow_complexity'] += 1
            self.generic_visit(node)

        def visit_Try(self, node):
            features['num_try_except'] += 1
            features['control_flow_complexity'] += 1 # Try block counts
            features['control_flow_complexity'] += len(node.handlers) # Each except counts
            if node.finalbody:
                 features['control_flow_complexity'] += 1 # Finally counts
            self.generic_visit(node)

    extractor = FeatureExtractor()
    extractor.visit(tree)

    features['max_nesting_depth'] = extractor.max_depth
    if extractor.function_lengths:
        features['avg_function_length'] = sum(extractor.function_lengths) / len(extractor.function_lengths)

    # Convert set to list for easier handling later if needed
    features['import_categories'] = list(features.get('import_categories', set()))

    return features

# --- Feature Similarity Calculation (Unchanged from your version) ---
def calculate_feature_similarity(features1: Dict, features2: Dict) -> float:
    """
    Calculate similarity based on structural features.
    Returns 0.0 if projects are fundamentally different.
    Adjusted thresholds.
    """
    if not features1 or not features2:
        return 0.0 # No features, no similarity

    # Check import categories - stronger penalty for mismatch
    cats1 = set(features1.get('import_categories', []))
    cats2 = set(features2.get('import_categories', []))

    # Avoid division by zero if both sets are empty
    if not cats1 and not cats2:
        category_overlap = 1.0 # Both use no categorized imports -> similar in that aspect
    elif not cats1 or not cats2:
        category_overlap = 0.0 # One uses imports, the other doesn't -> different
    else:
        # Jaccard index for category overlap
        intersection = len(cats1 & cats2)
        union = len(cats1 | cats2)
        category_overlap = intersection / union if union > 0 else 1.0

    # If domain overlap is low, cap overall feature similarity
    # Increased penalty: if less than 25% overlap, max 15% similarity
    if category_overlap < 0.25:
        return 0.15

    # Calculate numeric feature similarity (more robust calculation)
    numeric_features = [
        'num_functions', 'num_classes', 'num_imports', 'num_loops',
        'num_conditionals', 'num_try_except', 'max_nesting_depth',
        'avg_function_length', 'control_flow_complexity'
    ]

    similarities = []
    weights = { # Give slightly more weight to complexity, functions, classes
        'num_functions': 1.2, 'num_classes': 1.2, 'num_imports': 1.0,
        'num_loops': 1.0, 'num_conditionals': 1.0, 'num_try_except': 0.8,
        'max_nesting_depth': 1.1, 'avg_function_length': 1.0,
        'control_flow_complexity': 1.2
    }
    total_weight = 0.0

    for feature in numeric_features:
        val1 = features1.get(feature, 0)
        val2 = features2.get(feature, 0)

        # Handle zero cases
        if val1 == 0 and val2 == 0:
            sim = 1.0 # Both are zero -> similar in this aspect
        elif val1 == 0 or val2 == 0:
            sim = 0.0 # One is zero, the other isn't -> dissimilar
        else:
            # Normalized difference similarity
            diff = abs(val1 - val2)
            # Normalize by average or max? Average might be better
            # Avoid division by zero
            avg_val = (val1 + val2) / 2.0
            sim = max(0.0, 1.0 - (diff / avg_val)) # Ensure sim is not negative

        weight = weights.get(feature, 1.0)
        similarities.append(sim * weight)
        total_weight += weight

    if total_weight == 0: # Should not happen if there are features
         return category_overlap # Fallback to category overlap

    # Weighted average of numeric similarities, influenced by category overlap
    numeric_sim_avg = sum(similarities) / total_weight
    # Final feature similarity is a blend
    final_feature_sim = (numeric_sim_avg * 0.7) + (category_overlap * 0.3)

    return final_feature_sim


# --- Updated Normalization Functions ---
def normalize_python(code: str) -> Tuple[str, Dict]:
    """
    Enhanced normalization that preserves semantic structure.
    Returns: (normalized_code_string, structural_features_dict)
    """
    try:
        tree = ast.parse(code)

        # 1. Extract features *before* aggressive normalization modifies names too much
        features = extract_structural_features(code)

        # 2. Apply smart normalization (modifies tree in place)
        normalizer = SmartPythonNormalizer()
        normalized_tree = normalizer.visit(tree) # Pass the original tree

        # 3. Add more features extracted during normalization
        features['imports_found'] = list(normalizer.import_modules)
        features['api_calls_found'] = list(normalizer.api_calls)
        # Use a simple hash or signature of control flow
        control_flow_sig = ' '.join(normalizer.control_flow_pattern[:30]) # Limit length
        features['control_flow_signature_hash'] = hashlib.sha256(control_flow_sig.encode()).hexdigest()[:16]

        # 4. Generate normalized code string from the *modified* tree
        normalized_code = ast.unparse(normalized_tree)
        return normalized_code, features

    except Exception as e:
        print(f"Python AST processing failed: {e}")
        return "", {} # Return empty string and empty dict on failure

def normalize_java(code: str) -> Tuple[str, Dict]:
    """
    Parses and normalizes Java code using javalang.
    Returns: (normalized_token_string, empty_features_dict)
    """
    normalized_tokens = []
    try:
        tokens = javalang.tokenizer.tokenize(code)
        for token in tokens:
            if isinstance(token, javalang.tokenizer.Identifier):
                normalized_tokens.append("ID")
            # Keep keywords and operators
            elif isinstance(token, (javalang.tokenizer.Keyword, javalang.tokenizer.Operator, javalang.tokenizer.Separator, javalang.tokenizer.BasicType)):
                 normalized_tokens.append(token.value)
            # Maybe keep literals? Or replace with placeholder? Let's keep for now.
            elif isinstance(token, (javalang.tokenizer.Integer, javalang.tokenizer.DecimalInteger, javalang.tokenizer.FloatingPoint, javalang.tokenizer.String, javalang.tokenizer.Character)):
                 normalized_tokens.append("LITERAL") # Replace literals
            # Ignore comments and whitespace implicitly by not adding them
        
        # We don't have a feature extractor for Java yet
        return " ".join(normalized_tokens), {}
    except Exception as e:
        print(f"Java parsing failed: {e}")
        return "", {} # Return empty string and empty dict on failure


# --- Document Parsing (unchanged) ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts all text from a PDF given its bytes."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text("text", flags=fitz.TEXT_INHIBIT_SPACES) # Try to preserve structure a bit
            # Basic cleaning: replace multiple newlines/spaces carefully
            text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
            text = ' '.join(text.split()) # Consolidate spaces
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
            content_str = file_bytes.decode('latin-1', errors='ignore')
        except Exception as e:
            print(f"Text file decoding failed: {e}")
            return ""
     # Basic cleaning
    content_str = '\n'.join([line.strip() for line in content_str.splitlines() if line.strip()])
    return ' '.join(content_str.split()) # Consolidate spaces