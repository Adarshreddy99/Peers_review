# In: src/graph_combined.py

import os
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Use LangChain's HuggingFace integration
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import the other compiled graphs
from src.graph_plagiarism import plagiarism_app, PlagiarismState # Import state too
from src.graph_ai_ta import ai_ta_app, ReviewState # Import state too

# --- State Definition ---
class CombinedState(TypedDict):
    """Defines the state for the combined report graph."""
    # Inputs (shared with AI-TA)
    problem_statement: str
    code_files_input: List[dict] # {"name": str, "content": bytes}
    doc_files_input: List[dict]  # {"name": str, "content": bytes}

    # Intermediate Results
    plagiarism_report: str # Output from Plagiarism Graph (AI check only)
    ai_ta_feedback: str    # Output from AI-TA Graph

    # Final Output
    final_combined_report: str

# --- LangChain Chat Model Setup (Reused from other graphs) ---
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=2048, # Needs more room for combined report
        temperature=0.1
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)
    parser = StrOutputParser()
except Exception as e:
    print(f"Failed to initialize LangChain ChatHuggingFace model: {e}")
    chat_model = None
    parser = None

def build_mistral_chat_prompt(user_message: str) -> List[tuple[str, str]]:
     return [("user", f"[INST] {user_message.strip()} [/INST]")]

# --- Graph Nodes ---

def run_plagiarism_check_node(state: CombinedState) -> dict:
    """Invokes the plagiarism graph, forcing AI check only (no AST)."""
    print("---COMBINED: Running Plagiarism Check (AI Only)---")
    
    # Prepare input for the plagiarism graph
    files_for_plag = state.get("code_files_input", []) + state.get("doc_files_input", [])
    
    # Force AI check only, ignore AST setting from user (if it were present)
    plag_input_state = {
        "files_input": files_for_plag,
        "run_ast_check_input": False, # Explicitly disable AST for combined report
        "run_ai_check_input": True   # Always run AI check if possible
    }

    try:
        # Invoke the plagiarism graph
        plag_result_state = plagiarism_app.invoke(plag_input_state)
        report = plag_result_state.get("final_plagiarism_report", "Plagiarism check failed to generate report.")
        # We only want the AI part for the combined report context
        # Extract Semantic/AI section if present, otherwise use the full report
        ai_section = report 
        if "**Semantic & AI Analysis (LLM-based):**" in report:
            ai_section = report.split("**Semantic & AI Analysis (LLM-based):**", 1)[-1].split("\n\n**")[0].strip()
        elif "**Semantic & AI Analysis**" in report: # Handle slight variations
             ai_section = report.split("**Semantic & AI Analysis**", 1)[-1].split("\n\n**")[0].strip()

        # If the report indicates skipping due to file count, keep that message
        if "skipped because it only supports a maximum of 4 files" in report:
            ai_section = "AI plagiarism check was skipped (more than 4 files submitted)."
        elif "No plagiarism checks were selected or run" in report or "No files were uploaded" in report:
             ai_section = "No AI plagiarism check was performed (no files or check not applicable)."


    except Exception as e:
        print(f"Error invoking plagiarism graph: {e}")
        ai_section = f"Error during plagiarism check: {e}"

    return {"plagiarism_report": ai_section} # Store only the relevant part

def run_ai_ta_feedback_node(state: CombinedState) -> dict:
    """Invokes the AI-TA feedback graph, potentially adding plagiarism context."""
    print("---COMBINED: Running AI-TA Feedback---")

    # Prepare input for the AI-TA graph
    ai_ta_input_state = {
        "problem_statement": state.get("problem_statement", ""),
        "code_files_input": state.get("code_files_input", []),
        "doc_files_input": state.get("doc_files_input", [])
    }

    # Inject Plagiarism Context (Optional but helpful)
    # We can add the concise plagiarism findings to the problem statement
    # This gives the feedback LLM awareness of potential integrity issues.
    plag_context = state.get("plagiarism_report", "")
    if plag_context and "Error" not in plag_context and "No AI plagiarism check" not in plag_context:
        ai_ta_input_state["problem_statement"] += (
            f"\n\n**Academic Integrity Note:** An automated check found the following "
            f"regarding potential plagiarism or AI generation:\n{plag_context}"
        )

    try:
        # Invoke the AI-TA graph
        ai_ta_result_state = ai_ta_app.invoke(ai_ta_input_state)
        feedback = ai_ta_result_state.get("final_review", "Feedback generation failed.")
    except Exception as e:
        print(f"Error invoking AI-TA graph: {e}")
        feedback = f"Error during feedback generation: {e}"

    return {"ai_ta_feedback": feedback}


def synthesize_combined_report_node(state: CombinedState) -> dict:
    """Combines the plagiarism findings and AI-TA feedback into a final report."""
    print("---COMBINED: Synthesizing Final Report---")
    if not chat_model or not parser:
        return {"final_combined_report": "Error: LangChain Chat Model not initialized."}

    plag_report = state.get("plagiarism_report", "Plagiarism check information not available.")
    feedback = state.get("ai_ta_feedback", "Feedback not available.")

    # --- Prompt for Final LLM Combination ---
    user_prompt = (
        "You are a Head Professor compiling a final student report. "
        "You have two inputs:\n"
        "1.  A concise **Plagiarism & AI Generation Analysis**.\n"
        "2.  A detailed **Feedback Report** on the student's work.\n\n"
        "Combine these into a single, unified report for the student. "
        "Structure the report clearly.\n\n"
        "**Formatting Instructions:**\n"
        "- Use **bold text** for main section titles: '**Academic Integrity Analysis**' and '**Feedback on Submission**'.\n"
        "- Do NOT use markdown headings (like '##').\n"
        "- Present the information professionally and constructively.\n"
        "- Ensure the final report flows logically.\n\n"
        f"**Input 1: Plagiarism & AI Generation Analysis:**\n{plag_report}\n\n"
        f"**Input 2: Feedback on Submission:**\n{feedback}\n\n"
        "Now, generate the final combined report."
    )

    try:
        chain = chat_model | parser
        messages = build_mistral_chat_prompt(user_prompt)
        combined_report = chain.invoke(messages)
        return {"final_combined_report": combined_report}
    except Exception as e:
        print(f"Error during final combined report synthesis: {e}")
        # Fallback if synthesis fails
        combined_report = (
             f"**Academic Integrity Analysis:**\n{plag_report}\n\n"
             f"**Feedback on Submission:**\n{feedback}\n\n"
             f"*Error during final report synthesis: {e}*"
        )
        return {"final_combined_report": combined_report}

# --- Graph Definition ---
def build_combined_graph():
    workflow = StateGraph(CombinedState)

    # Add nodes that run the sub-graphs and the final synthesis
    workflow.add_node("run_plagiarism", run_plagiarism_check_node)
    workflow.add_node("run_feedback", run_ai_ta_feedback_node)
    workflow.add_node("synthesize_combined", synthesize_combined_report_node)

    # Define the sequential flow
    workflow.set_entry_point("run_plagiarism")
    workflow.add_edge("run_plagiarism", "run_feedback")
    workflow.add_edge("run_feedback", "synthesize_combined")
    workflow.add_edge("synthesize_combined", END)

    # Compile the graph
    combined_app = workflow.compile()
    return combined_app

# --- Export the compiled graph ---
combined_app = build_combined_graph()