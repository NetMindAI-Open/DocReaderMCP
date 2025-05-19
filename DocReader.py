#!/usr/bin/env python3
import os
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Dict, Any
from openai import OpenAI

from dotenv import load_dotenv
from fastmcp import FastMCP
from prompts_en import *

load_dotenv()
api_key = os.getenv('API_KEY')


client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",
    api_key=api_key,
)

# Create a FastMCP instance for the MCP Client
mcp = FastMCP(CLIENT_DESCRIPTION)

# 存储会话级别的搜索历史
search_history = []

def extract_doc_links(base_url: str, max_depth: int = 1) -> List[Tuple[str, str]]:
    """
    Extract document links and page titles from the base URL.
    :param base_url: The starting URL to crawl.
    :param max_depth: Maximum crawl depth (default 1).
    :return: List of found document links and titles [(url, title)]
    """
    visited = set()
    to_visit = [(base_url, 0)]
    doc_links = []
    
    while to_visit:
        url, depth = to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            visited.add(url)
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                # 只处理同域名的链接
                if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                    # 文档链接一般会包含特定关键词或路径，例如docs或document
                    if 'docs' in absolute_url.lower() or 'document' in absolute_url.lower():
                        link_title = link.get_text().strip() or absolute_url
                        doc_links.append((absolute_url, link_title))
                    elif depth < max_depth:
                        to_visit.append((absolute_url, depth + 1))
                    # 限制文档链接数量
                    if len(doc_links) > 100:
                        break
                        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")
            
    return doc_links

def find_most_relevant_page(pages: List[Tuple[str, str]], prompt: str, max_docs: int = 3) -> List[str]:
    """
    Use LLM to find the most relevant pages to the user's prompt.
    :param pages: List of pages [(url, title)]
    :param prompt: User prompt
    :param max_docs: Maximum number of documents to return (default 3)
    :return: List of URLs of the most relevant pages
    """
    # Format page info for LLM
    page_info = "\n".join([f"URL: {url}\nTitle: {title}" for url, title in pages])
    # Construct LLM prompt
    llm_prompt = PROMPT_FIND_RELEVANT_PAGE.format(max_docs=max_docs, prompt=prompt, page_info=page_info)
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {"role": "system", "content": SYSTEM_FIND_RELEVANT_PAGE},
            {"role": "user", "content": llm_prompt}
        ],
        temperature=0.3
    )
    
    # return the most relevant page URLs
    content = response.choices[0].message.content if response and response.choices else ""
    return content.strip().splitlines()[:max_docs] if content else []

def extract_page_content(url: str) -> str:
    """
    Extract content from a web page.
    :param url: Page URL
    :return: Extracted page content
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("div", class_="documentation")
        
        if main_content:
            page_content = main_content.get_text(strip=True)
        elif soup.body:
            page_content = soup.body.get_text(strip=True)
        else:
            page_content = ""
        
        return page_content
    except Exception as e:
        print(f"Error getting content from {url}: {str(e)}")
        return ""

# def get_page_links(url: str) -> List[Tuple[str, str]]:
#     """
#     从页面中提取所有链接
#     :param url: 页面URL
#     :return: 链接和标题的列表 [(url, title)]
#     """
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         links = []
        
#         for link in soup.find_all('a', href=True):
#             href = link['href']
#             absolute_url = urljoin(url, href)
#             link_title = link.get_text().strip() or absolute_url
#             links.append((absolute_url, link_title))
            
#         return links
#     except Exception as e:
#         print(f"提取页面 {url} 链接时出错: {str(e)}")
#         return []

def add_to_search_history(url: str, query: str, content: str) -> None:
    """Appending content to document search history."""
    global search_history
    # content_snippet = content[:200] + "..." if len(content) > 200 else content
    search_history.append({
        "url": url,
        "query": query,
        "content_snippet": content,
        "content": content
    })

# @mcp.tool()
def search_docs(
    doc_url: str, 
    query: str, 
    depth: int = 2, 
    max_results: int = 5
) -> List[Dict[str, str]]:
    """
    Search documentation pages and find the most relevant pages for the user's query.

    Args:
        doc_url: The documentation home or index page URL.
        query: The user's query or question.
        depth: Crawl depth (1-5).
        max_results: Maximum number of pages to return.

    Returns:
        A list of relevant pages, each containing URL and title.
    """
    doc_links = extract_doc_links(doc_url, max_depth=depth)
    
    if not doc_links:
        return []
    
    relevant_urls = find_most_relevant_page(doc_links, query, max_docs=max_results)
    
    result = []
    for url in relevant_urls:
        title = next((title for link, title in doc_links if link == url), url)
        result.append({"url": url, "title": title})
    
    return result

# @mcp.tool()
def extract_content(
    url: str, 
    query: str = ""
) -> Dict[str, Any]:
    """
    Extract content from the specified URL.

    Args:
        url: The page URL to extract content from.
        query: The user's query or question (optional, for history recording).

    Returns:
        A dictionary containing the page content, URL, and title.
    """
    content = extract_page_content(url)
    
    if not content:
        return {"url": url, "content": "", "success": False}
    
    # 保存到搜索历史
    add_to_search_history(url, query, content)
    
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else url
    except:
        title = url
    
    return {
        "url": url,
        "title": title,
        "content": content,
        "success": True,
        "length": len(content)
    }

# 工具3: 跟踪链接
# @mcp.tool()
# def follow_link(
#     source_url: str, 
#     link_pattern: str = "", 
#     max_links: int = 5
# ) -> List[Dict[str, str]]:
#     """
#     从源页面提取并跟踪链接
    
#     参数:
#     - source_url: 源页面URL
#     - link_pattern: 链接标题或URL中应包含的文本模式（可选）
#     - max_links: 返回的最大链接数
    
#     返回:
#     找到的链接列表，包含URL和标题
#     """
#     links = get_page_links(source_url)
    
#     if not links:
#         return []
    
#     # 如果提供了链接模式，过滤链接
#     filtered_links = []
#     if link_pattern:
#         for url, title in links:
#             if link_pattern.lower() in url.lower() or link_pattern.lower() in title.lower():
#                 filtered_links.append((url, title))
#     else:
#         filtered_links = links
    
#     # 限制结果数量
#     filtered_links = filtered_links[:max_links]
    
#     result = []
#     for url, title in filtered_links:
#         result.append({"url": url, "title": title})
    
#     return result

# 工具4: 总结发现
# @mcp.tool()
def summarize_findings(
    query: str
) -> Dict[str, Any]:
    """
    Finish user instruction based on collected information.
    :param query: The user's original query or question
    :return: LLM's response of the prompt according to the collected information
    """
    global search_history
    if not search_history:
        return {"summary": MSG_NO_INFO_TO_SUMMARIZE, "sources": []}
    # Chunk summary params
    MAX_CONTENT_LENGTH = 12000
    MAX_CHUNK_LENGTH = 3000
    def chunk_and_summarize(content, url, query):
        """If content is too long, chunk and summarize each part with respect to the query."""
        if len(content) <= MAX_CONTENT_LENGTH:
            return content
        # Chunk by paragraphs
        paragraphs = content.split('\n\n')
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < MAX_CHUNK_LENGTH:
                current += para + "\n\n"
            else:
                chunks.append(current)
                current = para + "\n\n"
        if current:
            chunks.append(current)

        summaries = []
        for idx, chunk in enumerate(chunks):
            prompt = PROMPT_SUMMARIZE_CHUNK.format(query=query, url=url, idx=idx+1, chunk=chunk)
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {"role": "system", "content": SYSTEM_SUMMARIZE_CHUNK},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            summary = response.choices[0].message.content if response and response.choices else ""
            summaries.append(summary)
        return "\n".join(summaries)
    # Build summary prompt
    content_blocks = []
    sources = []
    for entry in search_history:
        # Summarize each content block
        summarized_content = chunk_and_summarize(entry['content'], entry['url'], query)
        content_blocks.append(f"Source: {entry['url']}\nContent: {summarized_content}...")
        sources.append(entry['url'])
    combined_content = "\n\n".join(content_blocks)
    summary_prompt = PROMPT_FINAL_RESPONSE.format(query=query, combined_content=combined_content)
    # Use LLM to generate summary
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {"role": "system", "content": SYSTEM_FINAL_RESPONSE},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.3
    )
    summary = response.choices[0].message.content if response and response.choices else MSG_SUMMARY_FAILED
    return {
        "summary": summary,
        "sources": sources,
        "query": query
    }


@mcp.tool()
def read_doc(
    doc_url: str,
    query: str,
    depth: int = 2,
    max_results: int = 3
) -> Dict[str, Any]:
    """
    Complete the MCP workflow based on the document URL and user question, and return the final reply.
    :param doc_url: Document URL
    :param query: User question
    :param depth: Crawl depth (1-5)
    :param max_results: Number of documents to find (default 3)
    """
    results = search_docs(doc_url, query, depth=depth, max_results=max_results)
    if not results:
        return {"error": MSG_NOT_FOUND}
    for result in results:
        extract_content(result['url'], query)
    summary = summarize_findings(query)
    if not summary:
        return MSG_FINAL_SUMMARY_FAILED
    return summary['summary']
    

if __name__ == "__main__":

    # 测试read_doc
    result = read_doc(
        doc_url="https://huggingface.co/docs/transformers/en/model_doc/t5",
        query="如何利用T5来对于一个CSV数据文件进行监督微调？请写出完整代码。",
        depth=2,
        max_results=3
    )
    print(result)
    mcp.run()
