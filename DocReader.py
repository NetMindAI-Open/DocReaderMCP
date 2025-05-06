#!/usr/bin/env python3
import os
import sys
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from typing import List, Tuple
from openai import OpenAI
import warnings

from dotenv import load_dotenv

from fastmcp import FastMCP

load_dotenv()
api_key = os.getenv('API_KEY')


warnings.filterwarnings("ignore", message=".*PDF text extraction.*")

client = OpenAI(
    base_url="https://api.netmind.ai/inference-api/openai/v1",  # NetMind的API地址
    api_key=api_key,
)

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
    return response.choices[0].message.content.strip().splitlines()[:max_docs]

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
        else:
            page_content = soup.body.get_text(strip=True)
        
        return page_content
    except Exception as e:
        print(f"提取页面 {url} 内容时出错: {str(e)}")
        return ""

def process_page(urls: List[str], prompt: str) -> str:
    """
    处理多个页面并组合内容生成回答
    :param urls: 页面URL列表
    :param prompt: 用户提示
    :return: 生成的回答
    """
    try:
        all_content = []
        if isinstance(urls, str):
            urls = [urls]
        
        for url in urls:
            content = extract_page_content(url)
            if content:
                all_content.append(f"--- 来自 {url} 的内容 ---\n{content}")
        
        if not all_content:
            return "无法从提供的URL中提取有效内容"
        
        combined_content = "\n\n".join(all_content)
        return combined_content
    
    except Exception as e:
        return f"处理页面时出错: {str(e)}"


mcp = FastMCP("This tool can find relevant documents according to user prompt. You should provide a document main page URL and a user prompt. You can optionally set the depth of crawling and the maximum number of documents to retrieve.")

@mcp.tool()
def read_doc(url: str, user_prompt: str, depth: int = 5, max_pages: int = 3) -> str:
    """
    读取文档并回答问题
    :param url: 文档主页URL
    :param prompt: 用户问题
    :param depth: 爬取深度
    :param max_pages: 最大检索的相关文档数量
    :return: 生成的回答
    """
    base_url = url
    if not base_url.startswith(('http://', 'https://')):
        print("错误: 请输入有效的HTTP/HTTPS URL")
        sys.exit(1)
        
    doc_urls = extract_doc_links(base_url, depth)
    if doc_urls:
        relevant_urls = find_most_relevant_page(doc_urls, user_prompt, max_docs=max_pages)
        
        response = process_page(relevant_urls, user_prompt)
    else:
        response = process_page([url], user_prompt)
    return response




if __name__ == "__main__":
    mcp = FastMCP("This tool can read documents and answer questions based on them.")
    mcp.run()
    # print(read_doc("https://flax.readthedocs.io/en/latest/index.html", "如何用flax训练模型？", depth=5, max_pages=3))
