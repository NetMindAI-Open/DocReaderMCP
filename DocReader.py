#!/usr/bin/env python3
import os
import sys
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Tuple, Dict, Any
from openai import OpenAI
import warnings

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()
api_key = os.getenv('API_KEY')

warnings.filterwarnings("ignore", message=".*PDF text extraction.*")

client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",
    api_key=api_key,
)

# 创建FastMCP实例，提供详细描述
mcp = FastMCP("""
Document Reader MCP工具集 - 根据文档内容完成指定任务
这个工具集的工作流程是:
1. 从文档网站搜索相关页面
2. 提取特定页面内容
3. 深入探索链接，寻找与用户问题相关的内容
4. 根据文档内容最终完成用户指令

""")

# 存储会话级别的搜索历史
search_history = []

def extract_doc_links(base_url: str, max_depth: int = 1) -> List[Tuple[str, str]]:
    """
    从基础URL提取文档链接和页面标题
    :param base_url: 要爬取的起始URL
    :param max_depth: 最大爬取深度(默认为1)
    :return: 找到的文档链接和标题列表 [(url, title)]
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
            title = soup.title.string if soup.title else url
            
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
                        
        except Exception as e:
            print(f"爬取 {url} 时出错: {str(e)}")
            
    return doc_links

def find_most_relevant_page(pages: List[Tuple[str, str]], prompt: str, max_docs: int = 3) -> List[str]:
    """
    使用LLM找到与用户prompt最相关的页面
    :param pages: 页面列表 [(url, title)]
    :param prompt: 用户提示
    :param max_docs: 最大返回文档数量 (默认为3)
    :return: 最相关页面的URL的List
    """
    # 将页面信息格式化为LLM可理解的文本
    page_info = "\n".join([f"URL: {url}\nTitle: {title}" for url, title in pages])
    
    # 构造LLM提示
    llm_prompt = f"""请根据用户问题和文档页面信息，返回最相关的{max_docs}个页面的URL。
用户问题: {prompt}
文档页面信息:
{page_info}

请只返回最相关页面的URL，不要包含其他内容。每个URL单独一行。"""
    
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {"role": "system", "content": "你是一个文档助手，负责根据用户指令找到最相关的文档页面。你的回复内容**只能包含文档链接**，不能包含**任何其他内容**。每个链接单独一行。"},
            {"role": "user", "content": llm_prompt}
        ],
        temperature=0.3
    )
    
    # 返回最多max_docs个相关页面的URL
    content = response.choices[0].message.content if response and response.choices else ""
    return content.strip().splitlines()[:max_docs] if content else []

def extract_page_content(url: str) -> str:
    """
    从单个页面提取内容
    :param url: 页面URL
    :return: 提取的页面内容
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
        
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content") or soup.find("div", class_="documentation")
        
        # 如果找不到特定的内容区域，则使用整个body
        if main_content:
            page_content = main_content.get_text(strip=True)
        elif soup.body:
            page_content = soup.body.get_text(strip=True)
        else:
            page_content = ""
        
        return page_content
    except Exception as e:
        print(f"提取页面 {url} 内容时出错: {str(e)}")
        return ""

def get_page_links(url: str) -> List[Tuple[str, str]]:
    """
    从页面中提取所有链接
    :param url: 页面URL
    :return: 链接和标题的列表 [(url, title)]
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            link_title = link.get_text().strip() or absolute_url
            links.append((absolute_url, link_title))
            
        return links
    except Exception as e:
        print(f"提取页面 {url} 链接时出错: {str(e)}")
        return []

def add_to_search_history(url: str, query: str, content: str) -> None:
    """添加内容到搜索历史"""
    global search_history
    content_snippet = content[:200] + "..." if len(content) > 200 else content
    search_history.append({
        "url": url,
        "query": query,
        "content_snippet": content_snippet,
        "content": content
    })

# 工具1: 搜索文档
# @mcp.tool()
def search_docs(
    doc_url: str, 
    query: str, 
    depth: int = 2, 
    max_results: int = 5
) -> List[Dict[str, str]]:
    """
    搜索文档页面，找出与用户查询最相关的页面
    
    参数:
    - doc_url: 文档主页或索引页的URL
    - query: 用户查询或问题
    - depth: 爬取深度（1-5）
    - max_results: 返回的最大结果数量
    
    返回:
    相关页面列表，包含URL和标题
    """
    doc_links = extract_doc_links(doc_url, max_depth=depth)
    
    if not doc_links:
        return []
    
    relevant_urls = find_most_relevant_page(doc_links, query, max_docs=max_results)
    
    result = []
    for url in relevant_urls:
        # 找到对应的标题
        title = next((title for link, title in doc_links if link == url), url)
        result.append({"url": url, "title": title})
    
    return result

# 工具2: 提取内容
# @mcp.tool()
def extract_content(
    url: str, 
    query: str = ""
) -> Dict[str, Any]:
    """
    从指定URL提取页面内容
    
    参数:
    - url: 要提取内容的页面URL
    - query: 用户查询或问题（可选，用于记录历史）
    
    返回:
    包含页面内容、URL和标题的字典
    """
    content = extract_page_content(url)
    
    if not content:
        return {"url": url, "content": "", "success": False}
    
    # 保存到搜索历史
    add_to_search_history(url, query, content)
    
    # 尝试获取页面标题
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
def follow_link(
    source_url: str, 
    link_pattern: str = "", 
    max_links: int = 5
) -> List[Dict[str, str]]:
    """
    从源页面提取并跟踪链接
    
    参数:
    - source_url: 源页面URL
    - link_pattern: 链接标题或URL中应包含的文本模式（可选）
    - max_links: 返回的最大链接数
    
    返回:
    找到的链接列表，包含URL和标题
    """
    links = get_page_links(source_url)
    
    if not links:
        return []
    
    # 如果提供了链接模式，过滤链接
    filtered_links = []
    if link_pattern:
        for url, title in links:
            if link_pattern.lower() in url.lower() or link_pattern.lower() in title.lower():
                filtered_links.append((url, title))
    else:
        filtered_links = links
    
    # 限制结果数量
    filtered_links = filtered_links[:max_links]
    
    result = []
    for url, title in filtered_links:
        result.append({"url": url, "title": title})
    
    return result

# 工具4: 总结发现
# @mcp.tool()
def summarize_findings(
    query: str
) -> Dict[str, Any]:
    """
    根据已收集的信息总结发现
    
    参数:
    - query: 用户的原始查询或问题
    
    返回:
    总结信息，包含主要发现和参考的URL
    """
    global search_history
    
    if not search_history:
        return {"summary": "无可用信息进行总结", "sources": []}
    
    # 分块摘要参数
    MAX_CONTENT_LENGTH = 12000
    MAX_CHUNK_LENGTH = 3000
    
    def chunk_and_summarize(content, url, query):
        """如果内容过长则分块摘要，每块结合query做摘要"""
        if len(content) <= MAX_CONTENT_LENGTH:
            return content
        # 按段落分块
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
        # 针对性摘要
        summaries = []
        for idx, chunk in enumerate(chunks):
            prompt = f'请根据用户问题"{query}"对以下内容做摘要，只保留与问题最相关的信息（来源: {url}，第{idx+1}块）：\n{chunk}'
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {"role": "system", "content": "你是一个文档助手，只需根据用户问题对内容做针对性摘要，以避免信息冗余。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            summary = response.choices[0].message.content if response and response.choices else ""
            summaries.append(summary)
        return "\n".join(summaries)
    
    # 构建总结提示
    content_blocks = []
    sources = []
    
    for entry in search_history:
        # 针对每条内容做分块摘要
        summarized_content = chunk_and_summarize(entry['content'], entry['url'], query)
        content_blocks.append(f"来源: {entry['url']}\n内容: {summarized_content[:1000]}...")
        sources.append(entry['url'])
    
    combined_content = "\n\n".join(content_blocks)
    
    summary_prompt = f"""用户问题："{query}"\n\n 文档内容：\n\n{combined_content}\n\n\\n"""
    
    # 使用LLM生成总结
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3-0324",
        messages=[
            {"role": "system", "content": "你是一个根据文档信息解决用户问题的助手。请基于提供的文档内容，完成用户问题。"},
            {"role": "user", "content": summary_prompt}
        ],
        temperature=0.3
    )
    
    summary = response.choices[0].message.content if response and response.choices else "无法生成总结"
    
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
    根据文档URL和用户问题，完成MCP工作流程，返回最终回复

    参数:
    - doc_url: 文档URL
    - query: 用户问题
    - depth: 爬取深度（1-5）
    - max_results: 查找的文档数量，默认为3
    
    """
    results = search_docs(doc_url, query, depth=depth, max_results=max_results)
    if not results:
        return {"error": "未找到相关文档"}
    
    content = extract_content(results[0]['url'], query)
    if not content['success']:
        return {"error": "提取内容失败"}
    
    
    summary = summarize_findings(query)
    if not summary:
        return "error: 总结失败"
    
    # 5. 返回最终回复
    return summary['summary']
    
    
    
    

if __name__ == "__main__":

    # 测试read_doc
    # result = read_doc(
    #     doc_url="https://flax.readthedocs.io/en/latest/",
    #     query="如何用flax写一个LSTM网络？",
    #     depth=2,
    #     max_results=3
    # )
    # print(result)
    mcp.run()
