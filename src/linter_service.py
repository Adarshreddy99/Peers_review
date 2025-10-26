# In: src/linter_service.py

import subprocess
import os

# --- Configuration for Tool Paths ---
# Get the absolute path of the 'src' directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (one level up from 'src')
ROOT_DIR = os.path.dirname(SRC_DIR)
# Define paths to our bundled tools
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
CHECKSTYLE_JAR = os.path.join(TOOLS_DIR, "checkstyle-all.jar")
CHECKSTYLE_XML = os.path.join(TOOLS_DIR, "google_checks.xml")
# ------------------------------------

def run_linter(file_path: str, file_type: str) -> str:
    """
    Runs the appropriate linter on a given file and returns the output.
    
    Args:
        file_path: The absolute path to the temporary file to lint.
        file_type: The type of file ('py', 'java', 'c', 'cpp').

    Returns:
        A string containing the linter's raw output (stdout and stderr).
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    command = []

    if file_type == 'py':
        command = ["python", "-m", "flake8", file_path]
    
    elif file_type == 'java':
        if not os.path.exists(CHECKSTYLE_JAR) or not os.path.exists(CHECKSTYLE_XML):
            return "Error: Checkstyle JAR or XML not found in 'tools' directory."
        command = [
            "java",
            "-jar",
            CHECKSTYLE_JAR,
            "-c",
            CHECKSTYLE_XML,
            file_path
        ]
    
    elif file_type in ('c', 'cpp'):
        # cppcheck is a powerful linter for C/C++
        # We enable 'all' checks and '--inconclusive' to get more feedback
        command = [
            "cppcheck",
            "--enable=all",
            "--inconclusive",
            file_path
        ]
        
    else:
        return f"Linter not available for file type: {file_type}"

    try:
        # Run the command
        # We capture both stdout and stderr because different linters
        # report to different streams. cppcheck uses stderr.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30  # 30-second timeout
        )
        
        output = result.stdout + result.stderr
        
        if not output:
            return "Linter ran successfully. No issues found."
        
        return output
        
    except FileNotFoundError as e:
        # This happens if a tool (like 'flake8' or 'cppcheck') isn't installed
        return f"Error: Linter executable not found. Is it installed? \nDetails: {e}"
    except subprocess.TimeoutExpired:
        return "Error: Linter timed out (took longer than 30 seconds)."
    except Exception as e:
        return f"An unexpected error occurred while running the linter: {e}"