# In: ai-ta-platform/test_graph.py (Updated)

import os
from dotenv import load_dotenv

# --- CRITICAL: Load .env file BEFORE importing the graph ---
load_dotenv()
# -----------------------------------------------------------

from src.graph_plagiarism import plagiarism_app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join(BASE_DIR, "test_files")

def load_file_as_bytes(file_name: str) -> dict:
    """Loads a test file and returns the dict format the graph expects."""
    file_path = os.path.join(TEST_FILES_DIR, file_name)
    with open(file_path, 'rb') as f:
        content_bytes = f.read()
    return {"name": file_name, "content": content_bytes}

def run_test(name, files, run_ast, run_ai):
    """Helper function to run a single test case."""
    print(f"\n--- {name} ---")
    
    input_state = {
        "files_input": [load_file_as_bytes(f) for f in files],
        "run_ast_check_input": run_ast,
        "run_ai_check_input": run_ai
    }

    print(f"Invoking graph with {len(files)} file(s): "
          f"(AST: {run_ast}, AI: {run_ai})")
    
    final_state = plagiarism_app.invoke(input_state)
    
    print("\n--- FINAL REPORT ---")
    print(final_state["final_plagiarism_report"])
    print("----------------------")
    
    # Return the state for assertions
    return final_state


if __name__ == "__main__":
    
    # Test 1: 2 Python files, user wants both. Should run both.
    # Test 1: 2 Python files, user wants both. Should run both.
    state1 = run_test(
        "Test 1: 2 Py Files, AST=True, AI=True",
        files=["plag_a.py", "plag_b.py"],
        run_ast=True,
        run_ai=True
    )
    # Check that the main sections are present in the final summary
    assert "Structural Similarity" in state1["final_plagiarism_report"]
    assert "Semantic & AI Analysis" in state1["final_plagiarism_report"]

    # --- UPDATED ASSERTIONS ---
    # Check the RAW AST report in the state to ensure the percentage WAS calculated
    assert "100.0% similar" in state1["similarity_matrix_report"]

    # Check the FINAL summary for keywords indicating the similarity conclusion
    assert "identical" in state1["final_plagiarism_report"] or \
           "high degree of similarity" in state1["final_plagiarism_report"] or \
           "share the same core logic" in state1["final_plagiarism_report"]
    
    # Test 2: 1 file, user wants both. Should run AI ONLY.
    state2 = run_test(
        "Test 2: 1 File, AST=True, AI=True",
        files=["sample.py"],
        run_ast=True, # Should be ignored by router
        run_ai=True
    )
    assert "Structural Similarity" not in state2["final_plagiarism_report"]
    assert "Semantic & AI Analysis" in state2["final_plagiarism_report"]
    assert "AI generation" in state2["ai_llm_report"] # Check raw LLM output
    
    # Test 3: 5 files (incl. C++), user wants both. Should run AST ONLY (and skip C++).
    state3 = run_test(
        "Test 3: 5 Files, AST=True, AI=True",
        files=["plag_a.py", "plag_b.py", "sample.py", "sample.java", "sample.cpp"],
        run_ast=True,
        run_ai=True  # Should be ignored by router
    )
    assert "Structural Similarity" in state3["final_plagiarism_report"]
    assert "Semantic & AI Analysis" not in state3["final_plagiarism_report"]
    assert "Notice" in state3["final_plagiarism_report"] # AI skip notice
    assert "skipped" in state3["final_plagiarism_report"]
    assert "skipped: sample.cpp" in state3["similarity_matrix_report"] # Check AST skip
    

    print("\n--- ALL TESTS COMPLETED! ---")