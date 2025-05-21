# DocReader MCP Tool

DocReader is a powerful tool for reading and searching documents, built on the Model Context Protocol (MCP). It enables LLMs to search, extract, and synthesize information from web-based documents, assisting AI assistants in answering questions accordingly.

## Features

- Search for relevant pages across documentation websites
- Extract content from specific pages
- Aggregate and summarize discovered information
- Complete the document Q&A workflow in a single step

## Installation

### Requirements

- Python 3.7 or higher
- fastmcp
- beautifulsoup4
- requests
- openai
- python-dotenv

### Installation Steps

1. Clone or download this repository.

2. Install the required dependencies:

```bash
pip install fastmcp beautifulsoup4 requests openai python-dotenv
```

1. Create a `.env` file and add your API key, preferably a [NetMind](https://www.netmind.ai) API key:

```
API_KEY=your_api_key_here
```

## Usage

### Run Directly

```bash
cd path/to/DocReaderMCP
python DocReader.py
```

### Run with fastmcp CLI

```bash
cd path/to/DocReaderMCP
fastmcp run DocReader.py
```

### Using with Cursor

#### Method 1: Temporary Addition

1. In the Cursor interface, click the extensions/plugins icon in the left sidebar.
2. Locate the MCP section or select "Add Tool".
3. Choose "Add Local MCP Tool".
4. Enter a tool name, such as "DocReader".
5. Select the execution method (either point to the script path or connect via URL).

#### Method 2: Persistent Installation

```bash
cd path/to/DocReaderMCP
fastmcp install DocReader.py --name "DocReader" --with beautifulsoup4 requests openai python-dotenv
```

## Toolset

DocReader MCP provides the following tool functions:

1. **search_docs**: Search documentation pages to find those most relevant to your query.
2. **extract_content**: Extract content from a specified URL.
3. **summarize_findings**: Summarize the information collected.
4. **read_doc**: Complete the entire workflow—search, extraction, and summarization—in one step.


## Recommended Workflow

1. Start by using `search_docs` to find relevant pages on the documentation site.
2. Use `extract_content` to retrieve content from the most relevant pages.
3. Summarize your findings with `summarize_findings`.
4. Alternatively, use `read_doc` to perform all these steps at once.

## Example

See `test_doc_reader.py` for more examples of how to use each tool function.

A brief example:

```python
from DocReader import search_docs, extract_content, summarize_findings, read_doc

doc_url = "https://flax.readthedocs.io/en/latest/index.html"
query = "How do I train a model with flax? Please help me write the training code and the inference code after training."

# 1. Search for relevant pages
results = search_docs(doc_url, query, depth=2, max_results=3)

# 2. Extract content
if results:
    page_content = extract_content(results[0]['url'], query)

# 3. Summarize findings
summary = summarize_findings(query)
print(summary['summary'])

# 4. One-step workflow
final_answer = read_doc(doc_url, query)
print(final_answer)
```