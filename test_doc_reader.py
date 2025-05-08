#!/usr/bin/env python3
from DocReader import search_docs, extract_content, follow_link, summarize_findings

def test_doc_reader():
    """测试文档阅读工具流程"""
    # 1. 搜索相关页面
    print("Step 1: 搜索相关页面")
    url = "https://flax.readthedocs.io/en/latest/index.html"
    query = "如何用flax训练模型？请帮我写训练代码跟训练好了之后的推理代码"
    results = search_docs(url, query, depth=2, max_results=3)
    
    if results:
        print(f"找到 {len(results)} 个相关页面:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} - {result['url']}")
    else:
        print("未找到相关页面")
        return
    
    # 2. 提取内容
    print("\nStep 2: 提取页面内容")
    page_content = extract_content(results[0]['url'], query)
    
    if page_content['success']:
        print(f"成功提取 '{page_content['title']}' 的内容")
        print(f"内容长度: {page_content['length']} 字符")
        print(f"内容片段: {page_content['content'][:200]}...")
    else:
        print(f"无法提取内容，请检查URL是否可访问")
    
    # 3. 跟踪链接
    print("\nStep 3: 跟踪链接")
    links = follow_link(results[0]['url'], "train", max_links=3)
    
    if links:
        print(f"找到 {len(links)} 个包含 'train' 的链接:")
        for i, link in enumerate(links):
            print(f"{i+1}. {link['title']} - {link['url']}")
    else:
        print("未找到相关链接")
    
    # 4. 总结发现
    print("\nStep 4: 总结发现")
    summary = summarize_findings(query)
    
    print(f"总结信息:")
    print(summary['summary'])
    print("\n参考来源:")
    for source in summary['sources']:
        print(f"- {source}")

if __name__ == "__main__":
    test_doc_reader() 