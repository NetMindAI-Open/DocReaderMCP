# prompts_en.py
"""
Centralized English prompt and message templates for DocReaderMCP
"""

CLIENT_DESCRIPTION = """
DocReader MCP Client - Complete tasks according to document content.
Workflow:
1. Search relevant pages from documentation sites
2. Extract content from specific pages
3. (Optional) Abstract excessive content to avoid redundancy
4. Complete the user's instruction based on the document content
"""

SYSTEM_FIND_RELEVANT_PAGE = "You are a documentation assistant. Your reply must ONLY contain document links, and nothing else. Each link on a separate line."

PROMPT_FIND_RELEVANT_PAGE = (
    "Based on the user's question and the document page information, return the {max_docs} most relevant page URLs.\n"
    "User question: {prompt}\n"
    "Document page information:\n"
    "{page_info}\n\n"
    "Only return the most relevant page URLs, nothing else. Each URL on a separate line."
)

SYSTEM_SUMMARIZE_CHUNK = "You are a documentation assistant. Summarize the content according to the user's question, keeping only the most relevant information to avoid redundancy."

PROMPT_SUMMARIZE_CHUNK = (
    'Summarize the following content according to the user question:\n{query}\n\n Keeping only the most relevant information (Source: {url}, chunk {idx}):\n{chunk}'
)

SYSTEM_FINAL_RESPONSE = "You are an assistant who solves user questions based on documentation. Please answer the user's question based on the provided document content. You should use the same language as the user's question."

PROMPT_FINAL_RESPONSE = (
    'User question: "{query}"\n\nDocument content:\n\n{combined_content}\n\n'
)

MSG_NO_INFO_TO_SUMMARIZE = "No available information to summarize."
MSG_SUMMARY_FAILED = "Failed to generate summary."
MSG_NOT_FOUND = "No relevant documents found."
MSG_EXTRACTION_FAILED = "Content extraction failed."
MSG_FINAL_SUMMARY_FAILED = "error: Summary failed." 