# In: src/graph_plagiarism.py

import os
from typing import TypedDict, List, Optional, Dict # Added Dict
from langgraph.graph import StateGraph, END

# Use LangChain's HuggingFace integration
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import updated parsing functions
from src.parsing_utils import (
    normalize_python,
    normalize_java,
    calculate_feature_similarity, # Import the feature comparison function
    extract_text_from_pdf,
    read_text_file
)

# --- State Definition ---
class PlagiarismState(TypedDict):
    """Defines the state for the plagiarism graph."""
    # Inputs
    files_input: List[dict]
    run_ast_check_input: bool
    run_ai_check_input: bool

    # Internal state
    num_files: int
    code_files: List[dict]
    doc_files: List[dict]
    # Store normalized data AND features
    normalized_code_data: List[dict] # {"name": str, "normalized_content": str, "features": dict}
    ast_skipped_files: List[str]
    ai_check_skipped_message: str

    # Output reports (can be None)
    similarity_matrix_report: Optional[str] # Hybrid AST result
    ai_llm_report: Optional[str]

    # Final combined report
    final_plagiarism_report: str

# --- LangChain Chat Model Setup (Unchanged) ---
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=1536,
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

# entry_and_preprocess_node remains the same as your last version

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
                content_str = content_bytes.decode('latin-1', errors='ignore')
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
        "num_files": num_total_processed,
        "code_files": code_files,
        "doc_files": doc_files,
        "run_ast_check_input": do_ast_check,
        "run_ai_check_input": do_ai_check,
        "ai_check_skipped_message": ai_skipped_message,
        "ast_skipped_files": ast_skipped_files,
        "normalized_code_data": [] # Initialize
    }

def normalize_code_files(state: PlagiarismState) -> dict:
    """
    Parses (AST) compatible code files (Python, Java), normalizes them,
    and extracts structural features.
    """
    print("---PLAG: Normalizing Code Files & Extracting Features (AST)---")
    normalized_code_data = []

    for file_dict in state["code_files"]:
        name = file_dict["name"]
        content = file_dict["content"]
        normalized_content = ""
        features = {}

        if name.lower().endswith('.py'):
            normalized_content, features = normalize_python(content)
        elif name.lower().endswith('.java'):
            normalized_content, features = normalize_java(content) # Note: Java features is empty {}
        # Other code files (C, C++) are skipped

        normalized_code_data.append({
            "name": name,
            "normalized_content": normalized_content, # Will be "" for skipped files
            "features": features # Will be {} for Java/Skipped
        })

    return {"normalized_code_data": normalized_code_data}


def calculate_similarity(state: PlagiarismState) -> dict:
    """Calculates HYBRID similarity: combines feature sim and TF-IDF sim,
       giving more weight to the lower score to reduce false positives."""
    print("---PLAG: Calculating HYBRID Similarity (Lower Score Weighted)---")

    # Filter out files skipped or failed normalization/feature extraction
    valid_data = [
        item for item in state["normalized_code_data"]
        # Ensure BOTH normalized content AND features exist for Python
        # For Java, only normalized_content is needed for TF-IDF
        if item["normalized_content"] and (item["name"].lower().endswith('.py') and item["features"]) or (item["name"].lower().endswith('.java'))
    ]

    report_lines = ["**Structural Similarity (Code Only, Hybrid AST-based):**"]

    if len(valid_data) < 2:
        report_lines.append("Check requires at least 2 compatible code files (Python or Java) with successful parsing. No comparison was run.")
    else:
        try:
            # --- TF-IDF Calculation ---
            tfidf_file_names = [item["name"] for item in valid_data]
            tfidf_contents = [item["normalized_content"] for item in valid_data]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(tfidf_contents)
            cosine_sim_matrix = cosine_similarity(tfidf_matrix)
            # --------------------------

            # --- Feature Calculation & Hybrid Score ---
            threshold_high = 0.70 # Flag pairs above 70% combined sim
            threshold_moderate = 0.50 # Report pairs above 50% combined sim
            reported_pairs = set()

            for i in range(len(valid_data)):
                for j in range(i + 1, len(valid_data)):
                    file1_data = valid_data[i]
                    file2_data = valid_data[j]
                    pair = tuple(sorted((file1_data["name"], file2_data["name"])))

                    if pair in reported_pairs:
                        continue

                    # Get TF-IDF score
                    tfidf_sim = cosine_sim_matrix[i, j]

                    # Get Feature score (handle Python vs Java etc.)
                    feature_sim = 0.0
                    can_compare_features = (file1_data["name"].lower().endswith('.py') and
                                            file2_data["name"].lower().endswith('.py') and
                                            file1_data["features"] and file2_data["features"])

                    if can_compare_features:
                        feature_sim = calculate_feature_similarity(
                            file1_data["features"], file2_data["features"]
                        )
                    # For Java vs Java, or Py vs Java, use TF-IDF as the feature score proxy
                    # This means feature_sim will equal tfidf_sim in these cases
                    elif (file1_data["name"].lower().endswith('.java') and file2_data["name"].lower().endswith('.java')) or \
                         (file1_data["name"].lower().endswith('.py') != file2_data["name"].lower().endswith('.py')):
                        feature_sim = tfidf_sim
                    else: # Fallback if features missing for some reason
                         feature_sim = tfidf_sim


                    # --- NEW HYBRID LOGIC (Weight lower score more) ---
                    min_sim = min(tfidf_sim, feature_sim)
                    max_sim = max(tfidf_sim, feature_sim)

                    # Example weights: 70% weight to the minimum score, 30% to the maximum
                    combined_sim = (min_sim * 0.7) + (max_sim * 0.3)
                    # Ensure score is within [0, 1] bounds
                    combined_sim = max(0.0, min(1.0, combined_sim))
                    # -----------------------------------------------

                    # --- Reporting ---
                    report_line = f"- `{pair[0]}` and `{pair[1]}`: **{combined_sim*100:.1f}%** combined similarity "
                    details = []
                    if can_compare_features:
                         # Show both underlying scores when features were compared
                         details.append(f"(Features: {feature_sim*100:.1f}%)")
                         details.append(f"(Structure: {tfidf_sim*100:.1f}%)")
                    else:
                        # Otherwise just show the TF-IDF structure score which drove the result
                         details.append(f"(Structure: {tfidf_sim*100:.1f}%)")

                    report_line += " ".join(details)

                    if combined_sim >= threshold_high:
                        report_line += " (HIGH - Suspiciously similar)"
                    elif combined_sim >= threshold_moderate:
                         report_line += " (Moderate similarity)"
                    # else: low similarity, don't report

                    # Only add line if combined similarity is moderate or high
                    if combined_sim >= threshold_moderate:
                        report_lines.append(report_line)

                    reported_pairs.add(pair)
            # --- End Loop ---

            if len(report_lines) == 1: # Only header line was added
                report_lines.append("\nNo file pairs found with moderate or high structural similarity.")

        except Exception as e:
            report_lines.append(f"Error during similarity calculation: {e}")

    # Add note about any skipped files (C, C++, etc.)
    if state["ast_skipped_files"]:
        skipped_str = ", ".join(state["ast_skipped_files"])
        report_lines.append(f"\n*Note: Files not supported by AST were skipped: {skipped_str}*")

    return {"similarity_matrix_report": "\n".join(report_lines)}


# check_ai_plagiarism node remains the same as your last version
def check_ai_plagiarism(state: PlagiarismState) -> dict:
    """Uses the LLM for semantic comparison with detailed analysis and percentages."""
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
        content_preview = file_dict['content'][:3000] # Limit context size
        if len(file_dict['content']) > 3000:
             content_preview += "\n... [Content Truncated]"
        file_prompt_section += f"```\n{content_preview}\n```\n\n"

    # --- Enhanced Prompt for LLM with Percentage and Reasoning ---
    user_prompt = (
        "You are an expert academic integrity reviewer specializing in plagiarism detection. "
        f"You are given {len(all_files_for_ai)} file(s) (which may be code or text documents) to analyze.\n\n"
        "Your task is to detect **semantic plagiarism** and provide detailed analysis:\n\n"
        "**For Multiple Files (2-4 files):**\n"
        "1. Compare all files pairwise and identify semantic similarities\n"
        "2. **IMPORTANT - Catch These Specific Plagiarism Patterns:**\n"
        "   a) **Logic Equivalence**: Different syntax, same logic (e.g., `for` vs `while`, recursion vs iteration)\n"
        "   b) **Method/Function Rewriting**: Same algorithm, different implementation style (e.g., refactoring, list comps vs loops)\n"
        "   c) **Structural Variations**: Same solution, reorganized (e.g., functions split/merged, code reordered)\n"
        "   d) **Cosmetic Differences**: Variable renaming, comment changes (note these if logic is identical)\n"
        "   e) **Paraphrasing (Docs)**: Rewording ideas without citation.\n\n"
        "3. **Solution Space Consideration:**\n"
        "   - Is this problem very standard with few correct solutions (e.g., 'implement bubble sort')?\n"
        "   - If YES, state: '**Note**: Limited solution space; high similarity might be expected.' Adjust verdict accordingly.\n"
        "   - If NO (multiple approaches exist) and files use the SAME approach, suspect plagiarism.\n\n"
        "4. For EACH pair of files, provide:\n"
        "   - **Plagiarism Percentage**: Estimate semantic similarity (0-100%). Consider algorithm/idea similarity, penalize cosmetic changes less.\n"
        "   - **Specific Locations**: WHERE plagiarism occurs (function names, line ranges, paragraph sections).\n"
        "   - **Why It's Plagiarized**: SPECIFIC evidence (e.g., 'Both use identical logic in function X', 'Paragraph 2 heavily paraphrases File B Paragraph 1', 'Pattern detected: for-to-while conversion').\n"
        "   - **Verdict**: [Clear Plagiarism / Suspicious Similarity / Expected Similarity / No Plagiarism]\n\n"
        "**For Single File:**\n"
        "Analyze for signs of AI generation:\n"
        "   - **AI Generation Likelihood**: Percentage (0-100%)\n"
        "   - **Indicators**: List specific signs (generic language, perfect structure, lack of unique voice, unnatural formality, too complex/simple for task).\n\n"
        "**Output Format:**\n"
        "Use this structure clearly for each comparison/analysis:\n"
        "```\n"
        "**Analysis: [File A] vs [File B]** (or **Single File Analysis: [File Name]**)\n"
        "- Semantic Similarity / AI Likelihood: [X]%\n"
        "- Solution Space: [Limited / Multiple approaches possible / NA for single file]\n"
        "- Locations of Similarity / AI Indicators:\n"
        "  * [Specific location/indicator with details]\n"
        "- Reasons / Justification:\n"
        "  * [Detailed reason with evidence / pattern detected]\n"
        "- Verdict: [Clear Plagiarism / Suspicious / Expected / Likely AI / Likely Human / inconclusive]\n"
        "```\n\n"
        "Here are the files to analyze:\n"
        f"{file_prompt_section}"
        "\nProvide your detailed analysis now. Be specific, cite evidence, justify percentages, and consider the solution space."
    )

    try:
        chain = chat_model | parser
        messages = build_mistral_chat_prompt(user_prompt)
        ai_opinion = chain.invoke(messages)
        return {"ai_llm_report": ai_opinion}
    except Exception as e:
        print(f"Error during LangChain AI check invocation: {e}")
        return {"ai_llm_report": f"Error during AI check: {e}"}

# compile_plagiarism_report node remains the same as your last version
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
        "- Use clear paragraphs and bullet points for the content.\n"
        "- Preserve all percentage values and specific reasons for plagiarism."
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
        # Use titles in fallback
        if sim_report: report_parts.append(sim_report)
        if ai_report: report_parts.append(f"**Semantic & AI Analysis (LLM-based):**\n{ai_report}")
        if skipped_msg: report_parts.append(f"**Notice:**\n{skipped_msg}")
        final_report_str = "\n\n".join(report_parts).strip()
        final_report_str += f"\n\n*Error during final summary generation: {e}*"
        return {"final_plagiarism_report": final_report_str}


# --- Conditional Edges (Unchanged) ---
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

# --- Graph Definition (Unchanged) ---
def build_plagiarism_graph():
    workflow = StateGraph(PlagiarismState)

    workflow.add_node("entry_preprocess", entry_and_preprocess_node)
    workflow.add_node("normalize_code", normalize_code_files)
    workflow.add_node("calculate_similarity", calculate_similarity)
    workflow.add_node("check_ai_plagiarism", check_ai_plagiarism)
    workflow.add_node("compile_report", compile_plagiarism_report)

    workflow.set_entry_point("entry_preprocess")

    workflow.add_conditional_edges(
        "entry_preprocess",
        decide_ast_path,
        {
            "run_ast": "normalize_code",
            "skip_ast": "check_ai_plagiarism",
            "run_ai": "check_ai_plagiarism",
            "skip_ai": "compile_report"
        }
    )

    workflow.add_edge("normalize_code", "calculate_similarity")
    workflow.add_conditional_edges(
        "calculate_similarity",
        decide_ai_path,
        {
            "run_ai": "check_ai_plagiarism",
            "skip_ai": "compile_report"
        }
    )

    workflow.add_edge("check_ai_plagiarism", "compile_report")
    workflow.add_edge("compile_report", END)

    plagiarism_app = workflow.compile()
    return plagiarism_app

# --- This allows app.py to import the compiled graph ---
plagiarism_app = build_plagiarism_graph()