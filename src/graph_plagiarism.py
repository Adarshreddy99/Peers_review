# In: src/graph_plagiarism.py

import os
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

# Use LangChain's HuggingFace integration
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import parsing functions
from src.parsing_utils import (
    normalize_python,
    normalize_java,
    extract_text_from_pdf,
    read_text_file
)

# --- State Definition ---
class PlagiarismState(TypedDict):
    """Defines the state for the plagiarism graph."""
    # Inputs
    files_input: List[dict]  # List of {"name": str, "content": bytes}
    run_ast_check_input: bool
    run_ai_check_input: bool

    # Internal state
    num_files: int
    code_files: List[dict] # {"name": str, "content": str}
    doc_files: List[dict]  # {"name": str, "content": str} - content is extracted text
    normalized_code: List[dict] # {"name": str, "normalized_content": str}
    ast_skipped_files: List[str] # List of code files skipped by AST
    ai_check_skipped_message: str # Message for user if AI is skipped

    # Output reports (can be None)
    similarity_matrix_report: Optional[str] # AST result
    ai_llm_report: Optional[str]            # AI result

    # Final combined report
    final_plagiarism_report: str

# --- LangChain Chat Model Setup ---
HF_TOKEN = os.environ.get("HF_TOKEN")
# Using the specific model your teammate used
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        # max_new_tokens=1024, # Can configure parameters here
    )
    # Using temperature=0 for more deterministic outputs
    chat_model = ChatHuggingFace(llm=llm_endpoint, temperature=0)
    parser = StrOutputParser()

except Exception as e:
    print(f"Failed to initialize LangChain ChatHuggingFace model: {e}")
    chat_model = None
    parser = None

def build_mistral_chat_prompt(user_message: str) -> List[tuple[str, str]]:
    """Formats messages for Mistral Instruct models via LangChain."""
    # Using the specific format expected by Mistral Instruct
    return [("user", f"[INST] {user_message.strip()} [/INST]")]


# --- Graph Nodes ---

def entry_and_preprocess_node(state: PlagiarismState) -> dict:
    """
    First node: Decodes files, separates code/docs, extracts text,
    and determines which checks are allowed and requested.
    """
    print("---PLAG: Entry & Preprocess---")
    num_files = len(state.get("files_input", []))
    wants_ast = state.get("run_ast_check_input", False)
    wants_ai = state.get("run_ai_check_input", False)

    do_ast_check = False
    do_ai_check = False
    ai_skipped_message = ""
    code_files = []
    doc_files = []
    ast_skipped_files = [] # Files not processable by AST

    # Separate and process files
    for f_input in state["files_input"]:
        name = f_input["name"]
        content_bytes = f_input["content"]
        content_str = ""
        is_code = False
        is_doc = False

        # Try decoding as text first (covers code files and .txt)
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_str = content_bytes.decode('latin-1')
            except Exception:
                print(f"Warning: Could not decode file {name}. Skipping.")
                continue # Skip file if decoding fails completely

        # Classify and process
        if name.lower().endswith(('.py', '.java', '.c', '.cpp', '.h')):
            is_code = True
            code_files.append({"name": name, "content": content_str})
            # Mark files not supported by our simple AST parsers
            if not name.lower().endswith(('.py', '.java')):
                ast_skipped_files.append(name)
        elif name.lower().endswith('.pdf'):
            is_doc = True
            extracted_text = extract_text_from_pdf(content_bytes)
            if extracted_text:
                doc_files.append({"name": name, "content": extracted_text})
            else:
                print(f"Warning: Failed to extract text from PDF {name}. Skipping.")
        elif name.lower().endswith('.txt'):
            is_doc = True
            # We already decoded it above
            doc_files.append({"name": name, "content": content_str})
        else:
            print(f"Warning: Skipping unsupported file type: {name}")

    num_total_processed = len(code_files) + len(doc_files)

    # --- Implement Plagiarism Logic ---
    if num_total_processed == 0:
        pass # No checks possible
    elif num_total_processed == 1:
        do_ast_check = False # AST requires >= 2 code files
        if wants_ai:
            do_ai_check = True
    elif 1 < num_total_processed <= 4:
        # Can run AI check on all processed files (code + docs)
        if wants_ai:
            do_ai_check = True
        # Can run AST check *only if* >= 2 compatible code files exist
        num_compatible_code = len([cf for cf in code_files if cf["name"].lower().endswith(('.py', '.java'))])
        if wants_ast and num_compatible_code >= 2:
            do_ast_check = True
    elif num_total_processed > 4:
        # Cannot run AI check
        do_ai_check = False
        if wants_ai:
            ai_skipped_message = "Notice: The LLM-based semantic check was skipped because it only supports a maximum of 4 files."
        # Can run AST check *only if* >= 2 compatible code files exist
        num_compatible_code = len([cf for cf in code_files if cf["name"].lower().endswith(('.py', '.java'))])
        if wants_ast and num_compatible_code >= 2:
            do_ast_check = True

    # Update the state
    return {
        "num_files": num_total_processed, # Based on processed files
        "code_files": code_files,
        "doc_files": doc_files,
        "run_ast_check_input": do_ast_check,  # Override with *actual* plan
        "run_ai_check_input": do_ai_check,   # Override with *actual* plan
        "ai_check_skipped_message": ai_skipped_message,
        "ast_skipped_files": ast_skipped_files,
        "normalized_code": [] # Initialize
    }


def normalize_code_files(state: PlagiarismState) -> dict:
    """
    Parses (AST) compatible code files (Python, Java) and normalizes them.
    """
    print("---PLAG: Normalizing Code Files (AST)---")
    normalized_code = []

    for file_dict in state["code_files"]:
        name = file_dict["name"]
        content = file_dict["content"]
        normalized_content = "" # Default to empty

        if name.lower().endswith('.py'):
            normalized_content = normalize_python(content)
        elif name.lower().endswith('.java'):
            normalized_content = normalize_java(content)
        # Other code files (C, C++) are skipped here but included in the list

        normalized_code.append({
            "name": name,
            "normalized_content": normalized_content # Will be "" for skipped files
        })

    return {"normalized_code": normalized_code}


def calculate_similarity(state: PlagiarismState) -> dict:
    """Calculates TF-IDF + Cosine Similarity on normalized code strings."""
    print("---PLAG: Calculating Similarity---")

    # Filter out files that were skipped or failed normalization
    valid_data = [
        item for item in state["normalized_code"]
        if item["normalized_content"] # Ensure normalized content exists
    ]

    # Initialize report content
    report_lines = ["**Structural Similarity (Code Only, AST-based):**"]

    if len(valid_data) < 2:
        report_lines.append("Check requires at least 2 compatible code files (Python or Java). No comparison was run.")
    else:
        try:
            file_names = [item["name"] for item in valid_data]
            file_contents = [item["normalized_content"] for item in valid_data]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(file_contents)
            cosine_sim_matrix = cosine_similarity(tfidf_matrix)

            threshold = 0.70 # 70%
            reported_pairs = set()

            for i in range(len(file_names)):
                for j in range(i + 1, len(file_names)):
                    sim = cosine_sim_matrix[i, j]
                    if sim >= threshold:
                        pair = tuple(sorted((file_names[i], file_names[j])))
                        if pair not in reported_pairs:
                            report_lines.append(
                                f"- `{pair[0]}` and `{pair[1]}` are **{sim*100:.1f}%** structurally similar."
                            )
                            reported_pairs.add(pair)

            if not reported_pairs:
                report_lines.append("No compatible code files found to be structurally similar above the threshold.")

        except Exception as e:
            report_lines.append(f"Error during similarity calculation: {e}")

    # Add note about any skipped files (C, C++, etc.)
    if state["ast_skipped_files"]:
        skipped_str = ", ".join(state["ast_skipped_files"])
        report_lines.append(f"\n*Note: Files not supported by AST were skipped: {skipped_str}*")

    return {"similarity_matrix_report": "\n".join(report_lines)}


def check_ai_plagiarism(state: PlagiarismState) -> dict:
    """Uses the LLM (ChatHuggingFace) for semantic comparison of ALL files (up to 4)."""
    print("---PLAG: Checking AI/Semantic Plagiarism---")
    if not chat_model or not parser:
        return {"ai_llm_report": "Error: LangChain Chat Model not initialized."}

    # Combine code and doc files for the prompt (up to 4 total)
    all_files_for_ai = state["code_files"] + state["doc_files"]
    if len(all_files_for_ai) > 4:
         return {"ai_llm_report": "Error: AI check skipped internally (more than 4 files)."} # Should be caught by router

    file_prompt_section = ""
    for i, file_dict in enumerate(all_files_for_ai):
        file_type = "Code" if file_dict in state["code_files"] else "Document"
        file_prompt_section += f"--- {file_type} File {i+1}: {file_dict['name']} ---\n"
        # Truncate long document content if necessary
        content_preview = file_dict['content'][:3000] # Limit context size
        if len(file_dict['content']) > 3000:
             content_preview += "\n... [Content Truncated]"
        file_prompt_section += f"```\n{content_preview}\n```\n\n"

    # --- Detailed Prompt for LLM ---
    user_prompt = (
        "You are an expert academic integrity reviewer. "
        f"You are given {len(all_files_for_ai)} file(s) (which may be code or text documents) to analyze.\n\n"
        "Your primary task is to detect **semantic plagiarism** between the files. Look for:\n"
        "1.  **Shared Ideas/Logic:** Do the files present the same core concepts, arguments, or algorithms, even with different wording or variable names?\n"
        "2.  **Paraphrasing without Citation:** (Especially for documents) Does one document seem to heavily paraphrase another without giving credit?\n"
        "3.  **Structural Similarities:** (For code) Are algorithms implemented similarly (e.g., swapping `for` for `while`)? Are functions/sections just reordered?\n"
        "4.  **AI Generation (if 1 file):** If only one file is provided, analyze it for signs of AI generation (generic language, perfect structure, lack of unique voice/style, overly complex for the task).\n\n"
        "Here are the files:\n"
        f"{file_prompt_section}"
        "Provide a **concise, professional summary** of your findings in a markdown list. Focus on actionable insights.\n"
        "- If you find plagiarism, state *which* files are involved and *specifically why* (e.g., 'Document A heavily paraphrases Document B's introduction', 'Code A and Code B implement the same sorting algorithm with minor variable changes').\n"
        "- If analyzing a single file for AI generation, give a clear likelihood assessment and justification.\n"
        "- If no significant issues are found, state that clearly."
    )

    try:
        # Use the LangChain model
        chain = chat_model | parser
        messages = build_mistral_chat_prompt(user_prompt)
        ai_opinion = chain.invoke(messages)

        return {"ai_llm_report": ai_opinion}
    except Exception as e:
        print(f"Error during LangChain AI check invocation: {e}")
        return {"ai_llm_report": f"Error during AI check: {e}"}

def compile_plagiarism_report(state: PlagiarismState) -> dict:
    """The final node. Compiles all reports into one master report using bold titles."""
    print("---PLAG: Compiling Final Report---")

    sim_report = state.get("similarity_matrix_report")
    ai_report = state.get("ai_llm_report")
    skipped_msg = state.get("ai_check_skipped_message")

    # Handle cases where no checks ran
    if not state["run_ast_check_input"] and not state["run_ai_check_input"]:
         return {"final_plagiarism_report": "No plagiarism checks were selected to run."}
    elif state["num_files"] == 0:
         return {"final_plagiarism_report": "No files were uploaded or processed."}

    # Fallback if chat model isn't available for final formatting
    if not chat_model or not parser:
        report_parts = []
        if sim_report: report_parts.append(sim_report) # Use raw report title
        if ai_report: report_parts.append(f"**Semantic & AI Analysis (LLM-based):**\n{ai_report}")
        if skipped_msg: report_parts.append(f"**Notice:**\n{skipped_msg}")
        return {"final_plagiarism_report": "\n\n".join(report_parts).strip()}

    # --- Prompt for Final LLM Summarization ---
    available_sections = []
    prompt_sections = [
        "You are an expert reviewer writing a final, summary report for an academic integrity check. "
        "Your task is to be **concise, readable, and professional.** "
        "Summarize the findings from the following raw analysis reports provided below."
        "\n\n**Formatting Instructions:**\n"
        "- Use **bold text** for section titles (e.g., '**Structural Similarity**').\n"
        "- Do NOT use markdown headings (like '##' or '###').\n"
        "- Use clear paragraphs and bullet points for the content."
    ]

    # Dynamically determine which sections the LLM should generate
    if state["run_ast_check_input"] and sim_report: # Include only if check was run AND produced output
        available_sections.append("- '**Structural Similarity (Code Only, AST-based)**'")
        prompt_sections.append(f"### Raw Structural Report:\n{sim_report}")

    if state["run_ai_check_input"] and ai_report: # Include only if check was run AND produced output
        available_sections.append("- '**Semantic & AI Analysis (LLM-based)**'")
        prompt_sections.append(f"### Raw Semantic & AI Report:\n{ai_report}")

    if skipped_msg:
        available_sections.append("- '**Notice**'")
        prompt_sections.append(f"### Raw Notice:\n{skipped_msg}")

    # Handle case where checks ran but produced no actionable report (e.g., error during AI call)
    if not available_sections:
         return {"final_plagiarism_report": "Plagiarism analysis completed, but no specific findings were reported (check for errors in previous steps)."}


    # Add the strict instructions about *which* sections to include
    prompt_sections.insert(1,
        "You must **only** generate sections for the reports in this list:\n" +
        "\n".join(available_sections) +
        "\n\nDo not invent any other sections. Do not add commentary beyond summarizing the reports."
    )

    prompt_sections.append(
        "\nNow, please generate the final, combined summary report using only the sections and formatting specified."
    )

    prompt = "\n\n".join(prompt_sections)
    # --------------------------------------------------

    try:
        # Use LangChain model for final formatting
        chain = chat_model | parser
        messages = build_mistral_chat_prompt(prompt)
        final_report = chain.invoke(messages)

        return {"final_plagiarism_report": final_report}
    except Exception as e:
        print(f"Error during LangChain final report synthesis: {e}")
        # Fallback if final synthesis fails
        report_parts = []
        if sim_report: report_parts.append(sim_report)
        if ai_report: report_parts.append(f"**Semantic & AI Analysis (LLM-based):**\n{ai_report}")
        if skipped_msg: report_parts.append(f"**Notice:**\n{skipped_msg}")
        final_report_str = "\n\n".join(report_parts).strip()
        final_report_str += f"\n\n*Error during final summary generation: {e}*"
        return {"final_plagiarism_report": final_report_str}


# --- Conditional Edges ---

def decide_ast_path(state: PlagiarismState) -> str:
    """Decides if we should run the AST check based on the actual plan."""
    if state["run_ast_check_input"]:
        print("Route: -> normalize_code_files")
        return "run_ast"
    else:
        # If not running AST, decide if we should run AI
        return decide_ai_path(state) # Chain decision

def decide_ai_path(state: PlagiarismState) -> str:
    """Decides if we should run the AI check based on the actual plan."""
    if state["run_ai_check_input"]:
        print("Route: -> check_ai_plagiarism")
        return "run_ai"
    else:
        print("Route: -> (skip ai) -> compile_plagiarism_report")
        return "skip_ai"

# --- Graph Definition ---

def build_plagiarism_graph():
    workflow = StateGraph(PlagiarismState)

    # Add nodes
    workflow.add_node("entry_preprocess", entry_and_preprocess_node)
    workflow.add_node("normalize_code", normalize_code_files)
    workflow.add_node("calculate_similarity", calculate_similarity)
    workflow.add_node("check_ai_plagiarism", check_ai_plagiarism)
    workflow.add_node("compile_report", compile_plagiarism_report)

    # Set entry point
    workflow.set_entry_point("entry_preprocess")

    # Routing from entry point
    workflow.add_conditional_edges(
        "entry_preprocess",
        decide_ast_path, # First, decide if AST runs
        {
            "run_ast": "normalize_code",
            "skip_ast": "check_ai_plagiarism", # If AST skipped, directly check AI condition
            "run_ai": "check_ai_plagiarism",   # If AST skipped AND AI should run
            "skip_ai": "compile_report"     # If AST skipped AND AI skipped
        }
    )

    # AST path
    workflow.add_edge("normalize_code", "calculate_similarity")
    # After AST calculation, decide if AI runs
    workflow.add_conditional_edges(
        "calculate_similarity",
        decide_ai_path,
        {
            "run_ai": "check_ai_plagiarism",
            "skip_ai": "compile_report"
        }
    )

    # AI path (either from skipping AST or after AST)
    workflow.add_edge("check_ai_plagiarism", "compile_report")

    # Final compilation node leads to end
    workflow.add_edge("compile_report", END)

    # Compile the graph
    plagiarism_app = workflow.compile()
    return plagiarism_app

# --- This allows app.py to import the compiled graph ---
plagiarism_app = build_plagiarism_graph()