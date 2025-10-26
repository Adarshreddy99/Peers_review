# In: app.py (Gradio Version)

import gradio as gr
import os
from dotenv import load_dotenv

# --- Load Environment Variables ---
if os.path.exists(".env"):
    load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# --- Import Compiled Graphs ---
# Wrap imports in try-except (important for Gradio launch)
graphs_imported = False
plagiarism_app = None
ai_ta_app = None
combined_app = None

try:
    if HF_TOKEN: # Check if the token is loaded
        from src.graph_plagiarism import plagiarism_app
        from src.graph_ai_ta import ai_ta_app
        from src.graph_combined import combined_app
        graphs_imported = True
    else:
        print("ERROR: HF_TOKEN not found. Graphs cannot be initialized.")

except ImportError as e:
    print(f"ERROR: Failed to import graph modules: {e}")
except Exception as e:
     print(f"ERROR: An error occurred during graph initialization: {e}")

# --- Helper Function to Process Gradio Files ---
def process_gradio_files(uploaded_files):
    """Converts Gradio File objects/list to the format graphs expect."""
    if not uploaded_files:
        return []
    
    # Gradio might return a single file object or a list
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
        
    files_input_list = []
    for temp_file in uploaded_files:
        try:
            # temp_file.name has the full path in Gradio, we just need the filename
            filename = os.path.basename(temp_file.name)
            # Read bytes from the temporary file path
            with open(temp_file.name, 'rb') as f:
                content_bytes = f.read()
            
            files_input_list.append({
                "name": filename,
                "content": content_bytes
            })
        except Exception as e:
            print(f"Error processing uploaded file {temp_file.name}: {e}")
            
    return files_input_list

# --- Functions to Wrap Graph Invocations ---

def run_plagiarism_check(files_list, run_ast, run_ai):
    """Wrapper for the plagiarism graph."""
    if not graphs_imported or not plagiarism_app:
        return "Error: Plagiarism check service not available."
    if not files_list:
        return "Please upload files to check."
        
    files_input = process_gradio_files(files_list)
    if not files_input:
         return "Error processing uploaded files."

    plag_input_state = {
        "files_input": files_input,
        "run_ast_check_input": run_ast,
        "run_ai_check_input": run_ai
    }
    
    try:
        # Show progress (Gradio handles this nicely)
        yield "Analyzing files for plagiarism... This may take a moment."
        plag_result_state = plagiarism_app.invoke(plag_input_state)
        yield plag_result_state.get("final_plagiarism_report", "Error generating report.")
    except Exception as e:
        yield f"An error occurred: {e}"

def run_ai_ta_feedback(problem_statement, code_files_list, doc_files_list):
    """Wrapper for the AI-TA feedback graph."""
    if not graphs_imported or not ai_ta_app:
        return "Error: Feedback service not available."
    if not problem_statement:
        return "Please enter your problem statement or goal."
    if not code_files_list and not doc_files_list:
        return "Please upload at least one code file or document."

    code_files_input = process_gradio_files(code_files_list)
    doc_files_input = process_gradio_files(doc_files_list)
    # Check if processing failed for all files
    if code_files_list and not code_files_input:
         return "Error processing code files."
    if doc_files_list and not doc_files_input:
         return "Error processing document files."


    aita_input_state = {
        "problem_statement": problem_statement,
        "code_files_input": code_files_input,
        "doc_files_input": doc_files_input
    }

    try:
        yield "Generating feedback... This may take some time depending on file size and web search."
        aita_result_state = ai_ta_app.invoke(aita_input_state)
        yield aita_result_state.get("final_review", "Error generating feedback.")
    except Exception as e:
        yield f"An error occurred: {e}"

def run_combined_report(problem_statement, code_files_list, doc_files_list):
    """Wrapper for the combined report graph."""
    if not graphs_imported or not combined_app:
        return "Error: Combined report service not available."
    if not problem_statement:
        return "Please enter your problem statement or goal."
    if not code_files_list and not doc_files_list:
        return "Please upload at least one code file or document."

    code_files_input = process_gradio_files(code_files_list)
    doc_files_input = process_gradio_files(doc_files_list)
    if code_files_list and not code_files_input:
         return "Error processing code files."
    if doc_files_list and not doc_files_input:
         return "Error processing document files."

    comb_input_state = {
        "problem_statement": problem_statement,
        "code_files_input": code_files_input,
        "doc_files_input": doc_files_input
    }

    try:
        yield "Generating combined report... This involves multiple checks and may take longer."
        comb_result_state = combined_app.invoke(comb_input_state)
        yield comb_result_state.get("final_combined_report", "Error generating combined report.")
    except Exception as e:
        yield f"An error occurred: {e}"


# --- Gradio Interface Definition using Blocks ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 AI-Powered Peer Review Platform")

    # Check for Initialization Errors
    if not graphs_imported or not HF_TOKEN:
         gr.Markdown("## ⚠️ Application Initialization Failed")
         gr.Markdown("Could not load API keys or initialize backend graphs. Please check logs and environment configuration.")
    else:
        # --- Define Tabs ---
        with gr.Tabs():
            # --- Tab 1: Plagiarism Checker ---
            with gr.TabItem("Plagiarism Checker"):
                gr.Markdown("Upload multiple files (code, PDF, TXT) to check for similarities.")
                with gr.Row():
                    plag_files_input = gr.File(
                        label="Upload Files (Code, PDF, TXT)",
                        file_count="multiple",
                        # Specify allowed types if desired, None allows all
                        # file_types=['.py', '.java', '.c', '.cpp', '.h', '.txt', '.pdf']
                    )
                with gr.Row():
                    plag_ast_check = gr.Checkbox(label="Run Structural Check (AST - for Python/Java Code, needs 2+)", value=True)
                    plag_ai_check = gr.Checkbox(label="Run Semantic / AI-Gen Check (LLM - max 4 files)", value=True)
                plag_button = gr.Button("Check Plagiarism")
                plag_output = gr.Markdown(label="Plagiarism Report") # Use Markdown for formatted output

                plag_button.click(
                    fn=run_plagiarism_check,
                    inputs=[plag_files_input, plag_ast_check, plag_ai_check],
                    outputs=plag_output
                )

            # --- Tab 2: AI-TA Feedback ---
            with gr.TabItem("AI-TA Feedback"):
                gr.Markdown("Get constructive feedback on your code and/or documents based on your goal.")
                aita_problem_input = gr.Textbox(label="1. Enter your Problem Statement or Goal (Required)", lines=3)
                with gr.Row():
                    aita_code_input = gr.File(label="2. Upload Code Files (Optional)", file_count="multiple") # file_types=['.py', '.java', '.c', '.cpp', '.h'])
                    aita_doc_input = gr.File(label="3. Upload Documents (Optional, PDF/TXT)", file_count="multiple") # file_types=['.pdf', '.txt'])
                aita_button = gr.Button("Generate Feedback")
                aita_output = gr.Markdown(label="Feedback Report")

                aita_button.click(
                    fn=run_ai_ta_feedback,
                    inputs=[aita_problem_input, aita_code_input, aita_doc_input],
                    outputs=aita_output
                )

            # --- Tab 3: Combined Report ---
            with gr.TabItem("Combined Report"):
                gr.Markdown("Get a unified report including AI-driven plagiarism analysis and feedback.")
                comb_problem_input = gr.Textbox(label="1. Enter your Problem Statement or Goal (Required)", lines=3)
                with gr.Row():
                    comb_code_input = gr.File(label="2. Upload Code Files (Optional)", file_count="multiple")
                    comb_doc_input = gr.File(label="3. Upload Documents (Optional, PDF/TXT)", file_count="multiple")
                comb_button = gr.Button("Generate Combined Report")
                comb_output = gr.Markdown(label="Combined Student Report")

                comb_button.click(
                    fn=run_combined_report,
                    inputs=[comb_problem_input, comb_code_input, comb_doc_input],
                    outputs=comb_output
                )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    if graphs_imported and HF_TOKEN:
        print("Launching Gradio App...")
        # Share=True creates a public link (useful for testing/sharing temporarily)
        # Set debug=True for more detailed error logs during development
        demo.launch(share=False, debug=True)
    else:
        print("ERROR: Cannot launch Gradio app due to initialization failures.")
        # Optionally, launch a simple error message app
        with gr.Blocks() as error_demo:
            gr.Markdown("# ⚠️ Application Initialization Failed")
            gr.Markdown("Could not load API keys or initialize backend graphs. Please check logs and environment configuration.")
        error_demo.launch()