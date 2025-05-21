# DocReader MCP 工具

DocReader 是一个强大的文档阅读和搜索工具，基于 Model Context Protocol (MCP) 实现。它能够从网页文档中搜索、提取和整合信息，帮助AI助手回答关于文档内容的问题。

## 功能特点

- 从文档网站搜索相关页面
- 提取特定页面内容
- 整合和总结发现的信息
- 一键完成文档问答工作流

## 安装

### 环境要求

- Python 3.7+
- fastmcp
- beautifulsoup4
- requests
- openai
- python-dotenv

### 安装步骤

1. 克隆或下载本仓库

2. 安装依赖：

```bash
pip install fastmcp beautifulsoup4 requests openai python-dotenv
```

3. 创建 `.env` 文件，添加 API 密钥：

```
API_KEY=your_api_key_here
```

## 使用方法

### 直接运行

```bash
cd path/to/DocReaderMCP
python DocReader.py
```

### 使用 fastmcp CLI 运行

```bash
cd path/to/DocReaderMCP
fastmcp run DocReader.py
```

### 测试功能

可以运行测试脚本验证功能：

```bash
python test_doc_reader.py
```

### 在 Cursor 中使用

#### 方法一：临时添加

1. 在 Cursor 界面中，点击左侧边栏中的扩展/插件图标
2. 找到 MCP 部分或"添加工具"选项
3. 选择"添加本地 MCP 工具"
4. 输入工具名称，如"DocReader"
5. 选择运行方式（指向脚本路径或连接 URL）

#### 方法二：持久化安装

```bash
cd path/to/DocReaderMCP
fastmcp install DocReader.py --name "文档阅读器" --with beautifulsoup4 requests openai python-dotenv
```

## 工具集

DocReader MCP 实现了以下工具函数：

1. **search_docs**：搜索文档页面，找出与用户查询最相关的页面
2. **extract_content**：从指定 URL 提取页面内容
3. **summarize_findings**：根据已收集的信息总结发现
4. **read_doc**：一键完成文档搜索、内容提取和总结的完整工作流

> ⚠️ 注意：`follow_link` 工具在当前版本未实现，相关描述和示例已移除。

## 推荐工作流

1. 首先使用 `search_docs` 搜索文档主页上的相关页面
2. 然后使用 `extract_content` 提取最相关页面的内容
3. 最后用 `summarize_findings` 总结所有发现
4. 或直接使用 `read_doc` 一步完成上述流程

## 示例

请参考 `test_doc_reader.py` 中的示例，了解如何调用各工具函数。

简要示例：

```python
from DocReader import search_docs, extract_content, summarize_findings, read_doc

doc_url = "https://flax.readthedocs.io/en/latest/index.html"
query = "如何用flax训练模型？请帮我写训练代码跟训练好了之后的推理代码"

# 1. 搜索相关页面
results = search_docs(doc_url, query, depth=2, max_results=3)

# 2. 提取内容
if results:
    page_content = extract_content(results[0]['url'], query)

# 3. 总结发现
summary = summarize_findings(query)
print(summary['summary'])

# 4. 一步式工作流
final_answer = read_doc(doc_url, query)
print(final_answer)
```