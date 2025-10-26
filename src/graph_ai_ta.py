# In: src/graph_ai_ta.py

import os
from typing import TypedDict, List, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END

# Use LangChain's HuggingFace integration and Tavily
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient

# Import parsing functions
from src.parsing_utils import extract_text_from_pdf, read_text_file

# --- State Definition ---
class ReviewState(TypedDict):
    """Defines the state for the AI-TA feedback graph."""
    # Inputs
    problem_statement: str
    code_files_input: List[dict] # {"name": str, "content": bytes}
    doc_files_input: List[dict]  # {"name": str, "content": bytes}

    # Processed content
    aggregated_code_text: str # Combined string of all code file contents
    aggregated_doc_text: str  # Combined string of all extracted doc text

    # Web search results (uses operator.add to append if called multiple times)
    web_search_results: Annotated[str, operator.add]

    # Final output
    final_review: str

# --- LangChain & Tavily Client Setup ---
HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY") # Needs to be in .env

LLM_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
try:
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=2048, # Increased for detailed feedback
        temperature=0.1 # Slightly creative but mostly factual
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)
    parser = StrOutputParser()
except Exception as e:
    print(f"Failed to initialize LangChain ChatHuggingFace model: {e}")
    chat_model = None
    parser = None

try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except Exception as e:
    print(f"Failed to initialize Tavily Client: {e}. Web search will be disabled.")
    tavily_client = None

def build_mistral_chat_prompt(user_message: str) -> List[tuple[str, str]]:
    """Formats messages for Mistral Instruct models via LangChain."""
    # Using the specific format expected by Mistral Instruct
    return [("user", f"[INST] {user_message.strip()} [/INST]")]

# --- Graph Nodes ---

def preprocess_files_node(state: ReviewState) -> dict:
    """
    Extracts text from documents and aggregates all file content.
    (No linting in this graph per requirements).
    """
    print("---AI-TA: Preprocessing Files---")
    code_contents = []
    doc_contents = []

    # Process Code Files (just read content)
    for f_input in state.get("code_files_input", []):
        name = f_input["name"]
        content_bytes = f_input["content"]
        content_str = ""
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_str = content_bytes.decode('latin-1')
            except Exception:
                print(f"Warning: Could not decode code file {name}. Skipping.")
                continue
        code_contents.append(f"--- Code File: {name} ---\n```\n{content_str}\n```")

    # Process Document Files (extract text)
    for f_input in state.get("doc_files_input", []):
        name = f_input["name"]
        content_bytes = f_input["content"]
        extracted_text = ""
        if name.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(content_bytes)
        elif name.lower().endswith('.txt'):
            extracted_text = read_text_file(content_bytes)
        else:
            print(f"Warning: Skipping unsupported document type: {name}")
            continue

        if extracted_text:
             # Truncate long documents
             preview = extracted_text[:4000]
             if len(extracted_text) > 4000:
                  preview += "\n... [Content Truncated]"
             doc_contents.append(f"--- Document: {name} ---\n{preview}")
        else:
            print(f"Warning: Failed to extract text from document {name}. Skipping.")


    return {
        "aggregated_code_text": "\n\n".join(code_contents),
        "aggregated_doc_text": "\n\n".join(doc_contents)
    }


def agentic_web_search_node(state: ReviewState) -> dict:
    """
    Uses the LLM to decide if a web search is needed based on docs/problem.
    If yes, runs Tavily search.
    """
    print("---AI-TA: Agentic Web Search Decision---")
    if not chat_model or not parser or not tavily_client:
        print("Web search skipped: Clients not initialized.")
        return {"web_search_results": "Web search was not available."}

    problem = state.get("problem_statement", "N/A")
    doc_text = state.get("aggregated_doc_text", "")

    # Only run search if there are documents to analyze
    if not doc_text:
        print("Web search skipped: No document text provided.")
        return {"web_search_results": "Web search skipped (no documents)."}

    # --- Prompt for LLM Tool Use Decision ---
    user_prompt = (
        "You are a research assistant reviewing documents. "
        f"The user's overall goal is: '{problem}'.\n"
        f"Here is the aggregated text from the document(s) (potentially truncated):\n{doc_text}\n\n"
        "Based *only* on the document text and the user's goal, do you need to search the web to:\n"
        "a) Verify specific facts or claims mentioned in the documents?\n"
        "b) Check if the information presented is up-to-date?\n"
        "c) Find definitions for technical terms used?\n\n"
        "If YES, respond *only* with the single, most important search query needed (be specific!).\n"
        "If NO, respond *only* with the exact text: NO_SEARCH_NEEDED"
    )

    try:
        chain = chat_model | parser
        messages = build_mistral_chat_prompt(user_prompt)
        llm_decision = chain.invoke(messages).strip()

        print(f"LLM Search Decision: {llm_decision}")

        if "NO_SEARCH_NEEDED" in llm_decision or not llm_decision:
            return {"web_search_results": "Web search deemed unnecessary by AI."}
        else:
            # Run Tavily Search
            print(f"Running Tavily search for: {llm_decision}")
            search_response = tavily_client.search(query=llm_decision, search_depth="basic") # Use "basic" for speed
            
            # Format results concisely
            results_str = f"Web Search Results for query '{llm_decision}':\n"
            if search_response.get("results"):
                 for res in search_response["results"][:3]: # Top 3 results
                      results_str += f"- {res.get('title', 'N/A')}: {res.get('content', 'N/A')[:200]}...\n" # Snippet preview
            else:
                 results_str += "No relevant results found."

            return {"web_search_results": results_str}

    except Exception as e:
        print(f"Error during agentic web search: {e}")
        return {"web_search_results": f"Error during web search: {e}"}


def final_feedback_node(state: ReviewState) -> dict:
    """Generates the final feedback using all gathered context with detailed grading."""
    print("---AI-TA: Generating Final Feedback---")
    if not chat_model or not parser:
        return {"final_review": "Error: LangChain Chat Model not initialized."}

    problem = state.get("problem_statement", "No problem statement provided.")
    code_text = state.get("aggregated_code_text", "")
    doc_text = state.get("aggregated_doc_text", "")
    web_results = state.get("web_search_results", "") # Will exist even if skipped

    # --- Build the Enhanced System Prompt ---
    system_prompt = """You are an AI grader and feedback generator for student submissions, which may be either text (project write-ups) or code.

For each submission, you must provide a comprehensive analysis covering:

1. **Relevance**: Assess whether the submission content is relevant to the assignment/question. State "Relevant", "Partially Relevant", or "Not Relevant" and explain why.

2. **Pros**: List key strengths of the submission. For documents: focus on clarity, logic, methodology, proper use of formulas/expressions, data presentation. For code: focus on correctness, structure, efficiency, good practices.

3. **Cons**: List key weaknesses or issues. For documents: identify incorrect formulas, flawed methodology, missing data, logical fallacies. For code: identify bugs, inefficiencies, poor practices, errors.

4. **Clarity**: Evaluate overall clarity. Rate as "High", "Medium", or "Low" and justify your rating.

5. **Technical Accuracy**: 
   - For documents: Verify correctness of formulas, mathematical expressions, numerical data, methodology, and theoretical foundations
   - For code: Check algorithm correctness, logic errors, implementation issues

6. **Methodology & Numbers**: 
   - Analyze the methodology used (if applicable)
   - Examine numerical data, calculations, and statistical measures
   - Verify if numbers/metrics make sense in context
   - Check if data presentation is appropriate

7. **Scope of Implementability**: Assess how practical and feasible the proposed solution/approach is in real-world scenarios.

8. **Improvements**: Provide specific, actionable suggestions for enhancement. Be detailed and constructive.

Generate your feedback in the following format:
---
**Relevance**: [Relevant/Partially Relevant/Not Relevant - with explanation]

**Pros**: 
- [List main strengths with details]

**Cons**: 
- [List main weaknesses with details]

**Clarity**: [High/Medium/Low - with justification]

**Technical Accuracy**:
- [Analysis of formulas, expressions, methodology, or code correctness]

**Methodology & Numbers**:
- [Detailed analysis of approach, numerical data, calculations if present]

**Scope of Implementability**: [Assessment of practical feasibility]

**Improvements**: 
- [Specific, actionable suggestions numbered 1, 2, 3, etc.]

**Overall Summary**: [3-5 sentences providing holistic feedback]
---

Be thorough, constructive, and detailed. Cover everything in the submission."""

    # --- Build the User Prompt ---
    prompt_sections = [
        f"**Assignment/Question**: {problem}\n"
    ]

    if code_text:
        prompt_sections.append(f"**Submitted Code**:\n{code_text}\n")

    if doc_text:
        prompt_sections.append(f"**Submitted Document(s)**:\n{doc_text}\n")

    # Include web results only if a search was actually performed and yielded results
    if web_results and "Error" not in web_results and "unnecessary" not in web_results and "skipped" not in web_results:
         prompt_sections.append(f"**Additional Web Context**:\n{web_results}\n")

    user_prompt = "\n".join(prompt_sections)

    try:
        # Use ChatPromptTemplate for proper system/user message handling
        feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{submission}")
        ])
        
        chain = feedback_prompt | chat_model | parser
        feedback = chain.invoke({"submission": user_prompt})
        return {"final_review": feedback}
    except Exception as e:
        print(f"Error during final feedback generation: {e}")
        return {"final_review": f"Error generating feedback: {e}"}


# --- Conditional Edge ---
def should_search_condition(state: ReviewState) -> str:
    """Decide whether to run the web search based on document presence."""
    if state.get("aggregated_doc_text") and tavily_client:
        print("Route: -> agentic_web_search")
        return "run_search"
    else:
        print("Route: -> (skip search) -> final_feedback")
        return "skip_search"

# --- Graph Definition ---
def build_ai_ta_graph():
    workflow = StateGraph(ReviewState)

    workflow.add_node("preprocess_files", preprocess_files_node)
    workflow.add_node("agentic_web_search", agentic_web_search_node)
    workflow.add_node("final_feedback", final_feedback_node)

    workflow.set_entry_point("preprocess_files")

    workflow.add_conditional_edges(
        "preprocess_files",
        should_search_condition,
        {
            "run_search": "agentic_web_search",
            "skip_search": "final_feedback" # Skip search if no docs or no client
        }
    )

    workflow.add_edge("agentic_web_search", "final_feedback")
    workflow.add_edge("final_feedback", END)

    ai_ta_app = workflow.compile()
    return ai_ta_app

# --- Export the compiled graph ---
ai_ta_app = build_ai_ta_graph()