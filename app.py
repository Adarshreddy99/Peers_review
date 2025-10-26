# In: app.py (Corrected Theme - Full Width Inputs)

import gradio as gr
import os
from dotenv import load_dotenv
import time # For slight delay simulation if needed

# --- Load Environment Variables ---
if os.path.exists(".env"):
    load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY") # Needed by AI-TA graph

# --- Import Compiled Graphs ---
graphs_imported = False
plagiarism_app = None
ai_ta_app = None
combined_app = None

try:
    if HF_TOKEN:
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
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]
    files_input_list = []
    for temp_file in uploaded_files:
        try:
            filename = os.path.basename(temp_file.name)
            # Gradio temp files need to be read from their path
            with open(temp_file.name, 'rb') as f:
                content_bytes = f.read()
            files_input_list.append({"name": filename, "content": content_bytes})
        except Exception as e:
            # Provide specific error if file path is missing (can happen with Gradio state)
            file_path = getattr(temp_file, 'name', 'N/A')
            print(f"Error processing uploaded file '{file_path}': {e}")
    return files_input_list


# --- Functions to Wrap Graph Invocations (with Progress) ---
def run_plagiarism_check(files_list, run_ast, run_ai, progress=gr.Progress(track_tqdm=True)):
    """Wrapper for the plagiarism graph with progress."""
    progress(0, desc="Starting Plagiarism Check...")
    if not graphs_imported or not plagiarism_app:
        return "**Error:** Plagiarism check service not available."
    if not files_list:
        return "**Warning:** Please upload files to check for plagiarism."

    progress(0.1, desc="Processing uploaded files...")
    files_input = process_gradio_files(files_list)
    if not files_input and files_list: # Check if processing failed
         return "**Error:** Could not process uploaded files."
    elif not files_input: # No files uploaded initially
        return "**Warning:** Please upload files to check for plagiarism."


    plag_input_state = {
        "files_input": files_input,
        "run_ast_check_input": run_ast,
        "run_ai_check_input": run_ai
    }

    report = "**Error:** Analysis failed." # Default error message
    try:
        progress(0.3, desc="Running plagiarism analysis...")
        # Invoke might take time
        plag_result_state = plagiarism_app.invoke(plag_input_state)
        progress(1.0, desc="Analysis Complete.")
        report = plag_result_state.get("final_plagiarism_report", "**Error:** Failed to generate report.")
    except Exception as e:
        progress(1.0)
        print(f"Plagiarism Check Error: {e}") # Log detailed error
        report = f"**An error occurred during analysis:** {str(e)}"

    return report # Return the final string report

def run_ai_ta_feedback(problem_statement, code_files_list, doc_files_list, progress=gr.Progress(track_tqdm=True)):
    """Wrapper for the AI-TA feedback graph with progress updates."""
    progress(0, desc="Starting Feedback Generation...")
    if not graphs_imported or not ai_ta_app:
        return "**Error:** Feedback service not available."
    if not problem_statement:
        return "**Warning:** Please enter problem statement."
    if not code_files_list and not doc_files_list:
        return "**Warning:** Please upload at least one file."

    progress(0.1, desc="Processing files...")
    code_files_input = process_gradio_files(code_files_list)
    doc_files_input = process_gradio_files(doc_files_list)

    if code_files_list and not code_files_input:
         return "**Error:** Processing code files failed."
    if doc_files_list and not doc_files_input:
         return "**Error:** Processing document files failed."

    aita_input_state = {
        "problem_statement": problem_statement,
        "code_files_input": code_files_input,
        "doc_files_input": doc_files_input
    }

    report = "**Error:** Feedback generation failed."
    try:
        progress(0.3, desc="Generating feedback (may include web search)...")
        aita_result_state = ai_ta_app.invoke(aita_input_state)
        progress(1.0, desc="Feedback Ready.")
        report = aita_result_state.get("final_review", "**Error:** Failed to generate feedback report.")
    except Exception as e:
        progress(1.0)
        print(f"AI-TA Feedback Error: {e}")
        report = f"**An error occurred during feedback generation:** {str(e)}"

    return report

def run_combined_report(problem_statement, code_files_list, doc_files_list, progress=gr.Progress(track_tqdm=True)):
    """Wrapper for the combined report graph with progress updates."""
    progress(0, desc="Starting Combined Report...")
    if not graphs_imported or not combined_app:
        return "**Error:** Combined report service not available."
    if not problem_statement:
        return "**Warning:** Please enter problem statement."
    if not code_files_list and not doc_files_list:
        return "**Warning:** Please upload at least one file."

    progress(0.1, desc="Processing files...")
    code_files_input = process_gradio_files(code_files_list)
    doc_files_input = process_gradio_files(doc_files_list)
    if code_files_list and not code_files_input:
         return "**Error:** Processing code files failed."
    if doc_files_list and not doc_files_input:
         return "**Error:** Processing document files failed."

    comb_input_state = {
        "problem_statement": problem_statement,
        "code_files_input": code_files_input,
        "doc_files_input": doc_files_input
    }

    report = "**Error:** Combined report generation failed."
    try:
        progress(0.3, desc="Running combined analysis (plagiarism & feedback)...")
        comb_result_state = combined_app.invoke(comb_input_state)
        progress(1.0, desc="Combined Report Generated.")
        report = comb_result_state.get("final_combined_report", "**Error:** Failed to generate combined report.")
    except Exception as e:
        progress(1.0)
        print(f"Combined Report Error: {e}")
        report = f"**An error occurred during combined report generation:** {str(e)}"

    return report

# --- Gradio Interface Definition using Blocks ---
# Use a custom dark theme with orange accents and Calibri font
# Use a custom dark theme with orange accents and Calibri font
custom_theme = gr.themes.Default(
    primary_hue="orange",
    secondary_hue="neutral",
    font=[gr.themes.GoogleFont("Calibri"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    # --- Dark Mode Overrides ---
    # Backgrounds
    body_background_fill_dark="#1A1A1A",
    block_background_fill_dark="#2A2A2A",
    input_background_fill_dark="#3A3A3A",

    # Text Colors
    body_text_color_dark="white",
    block_label_text_color_dark="white",
    block_title_text_color_dark="white",
    input_placeholder_color_dark="#A0A0A0",
    link_text_color_dark="#FFA500", # Orange links

    # Borders and Accents
    border_color_accent_dark="#FFA500",      # Orange border
    color_accent_soft_dark="#FFCC80",        # Lighter orange

    # Button Colors
    button_primary_background_fill_dark="#FFA500",
    button_primary_text_color_dark="black",
    button_secondary_background_fill_dark="#4A4A4A",
    button_secondary_text_color_dark="white",
)

with gr.Blocks(theme=custom_theme) as demo:

    gr.Markdown(
        """
        # AI-Powered Peer Review Platform
        *Get insights into academic integrity and receive constructive feedback.*
        """
    )
    gr.HTML("<hr style='border-color: #4A4A4A;'>") # Separator matching dark theme

    # Check for Initialization Errors at the top
    if not graphs_imported or not HF_TOKEN:
         gr.Markdown("## Application Initialization Failed")
         gr.Markdown(
             "**Error:** Could not load API keys or initialize backend graphs. "
             "Please ensure `HF_TOKEN` is set correctly and restart the application. Check console logs for details."
         )
    else:
        # --- Define Tabs ---
        with gr.Tabs():

            # --- Tab 1: Plagiarism Checker ---
            with gr.TabItem("Plagiarism Checker"):
                gr.Markdown("### Detect Structural and Semantic Plagiarism")
                gr.Markdown("Upload code (Python, Java), PDF, or TXT files. Select analysis types.")

                with gr.Column():
                    plag_files_input = gr.File(
                        label="Upload Files (Code, PDF, TXT)",
                        file_count="multiple",
                    )
                    with gr.Row():
                         plag_ast_check = gr.Checkbox(
                            label="Structural Check (AST)",
                            value=True,
                            info="Python/Java structure (needs 2+)"
                        )
                         plag_ai_check = gr.Checkbox(
                            label="Semantic Check (AI)",
                            value=True,
                            info="Deep analysis via LLM (max 4 files)"
                        )
                    plag_button = gr.Button("Check Plagiarism", variant="primary", size="lg")

                gr.HTML("<hr style='border-color: #4A4A4A;'>")
                with gr.Column():
                    plag_output = gr.Markdown(
                        label="Plagiarism Report",
                        value="*Upload files, select checks, and click button...*",
                    )

                plag_button.click(
                    fn=run_plagiarism_check,
                    inputs=[plag_files_input, plag_ast_check, plag_ai_check],
                    outputs=plag_output,
                    show_progress="full"
                )

            # --- Tab 2: AI-TA Feedback ---
            with gr.TabItem("AI-TA Feedback"):
                gr.Markdown("### Get AI-Powered Feedback")
                gr.Markdown("Receive detailed feedback on code and/or documents based on your goal. Web search may be used for documents.")

                with gr.Column():
                    aita_problem_input = gr.Textbox(
                        label="Problem Statement / Goal (Required)",
                        placeholder="Describe the assignment or project goal...",
                        lines=4
                    )
                    with gr.Row():
                         aita_code_input = gr.File(
                            label="Code Files (Optional)",
                            file_count="multiple"
                        )
                         aita_doc_input = gr.File(
                            label="Documents (Optional, PDF/TXT)",
                            file_count="multiple",
                            file_types=['.pdf', '.txt']
                        )
                    aita_button = gr.Button("Generate Feedback", variant="primary", size="lg")

                gr.HTML("<hr style='border-color: #4A4A4A;'>")
                with gr.Column():
                     aita_output = gr.Markdown(
                         label="Feedback Report",
                         value="*Enter goal and upload files to get feedback...*"
                     )

                aita_button.click(
                    fn=run_ai_ta_feedback,
                    inputs=[aita_problem_input, aita_code_input, aita_doc_input],
                    outputs=aita_output,
                    show_progress="full"
                )

            # --- Tab 3: Combined Report ---
            with gr.TabItem("Combined Report"):
                gr.Markdown("### Generate a Comprehensive Report")
                gr.Markdown("Get a unified report including AI-driven plagiarism analysis (semantic only) and feedback.")

                with gr.Column():
                    comb_problem_input = gr.Textbox(
                        label="Problem Statement / Goal (Required)",
                        placeholder="Describe the assignment or project goal...",
                        lines=4
                    )
                    with gr.Row():
                        comb_code_input = gr.File(
                            label="Code Files (Optional)",
                            file_count="multiple"
                        )
                        comb_doc_input = gr.File(
                            label="Documents (Optional, PDF/TXT)",
                            file_count="multiple",
                            file_types=['.pdf', '.txt']
                        )
                    comb_button = gr.Button("Generate Combined Report", variant="primary", size="lg")

                gr.HTML("<hr style='border-color: #4A4A4A;'>")
                with gr.Column():
                    comb_output = gr.Markdown(
                        label="Combined Report",
                        value="*Enter goal and upload files for the full report...*"
                    )

                comb_button.click(
                    fn=run_combined_report,
                    inputs=[comb_problem_input, comb_code_input, comb_doc_input],
                    outputs=comb_output,
                    show_progress="full"
                )

# --- Launch the Gradio App ---
if __name__ == "__main__":
    if graphs_imported and HF_TOKEN:
        print("Launching Gradio App (Dark Orange Theme)...")
        demo.launch(share=False, debug=True, show_error=True, inbrowser=True)
    else:
        print("ERROR: Cannot launch Gradio app due to initialization failures.")
        with gr.Blocks() as error_demo:
             gr.Markdown("## Application Initialization Failed")
             gr.Markdown(
                 "**Error:** Could not load API keys or initialize backend graphs. "
                 "Please ensure `HF_TOKEN` is set in your `.env` file and restart. Check console logs for details."
             )
        error_demo.launch(inbrowser=True)