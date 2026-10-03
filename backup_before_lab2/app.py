import os
import gradio as gr
# import spaces

from dotenv import load_dotenv
load_dotenv(".env")  # Load environment variables from .env file
HF_TOKEN = os.getenv("HF_TOKEN")

from smolagents import CodeAgent, InferenceClientModel
from tools import search_arxiv

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ.get("HF_TOKEN")
)

agent = CodeAgent(
    tools=[search_arxiv], 
    model=model,
    add_base_tools=False,
    max_steps=5
)

# # 1. THE DECOY: Passes the HF startup check so the container boots
# @spaces.GPU
# def decoy_function():
#     pass

def agent_chat(user_input):
    # 1. Run your existing agent logic here
    # ---------------------------------------------------------
    # Example: 
    # titles, authors = your_agent_search_function(user_input)
    # ---------------------------------------------------------
    
    # For demonstration, this is the tuple your agent was returning:
    titles = [
        'Towards trustworthy agentic AI: a comprehensive survey of safety, robustness, privacy, and system security', 
        'Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use'
    ]
    authors = [
        'Jinhu Qi, Muzhi Li, Jiahong Liu, Yuqin Shu, Dianzhi Yu, Shicheng Ma, Wenqian Cui, Yiyang Zhao, Yiyi Chen, Ruoxi Jiang, Irwin King, Zenglin Xu', 
        'Francisco Javier Arceo, Varsha Prasad Narsing'
    ]

    # 2. Format the tuple into a single Markdown-compatible string
    # ---------------------------------------------------------
    if not titles or not authors:
        return "No results found."

    markdown_output = "### 📚 Agent Search Results\n\n"
    
    for title, author in zip(titles, authors):
        # Replace unexpected newlines from the raw agent output to keep formatting clean
        clean_title = title.replace('\n', ' ').strip()
        clean_author = author.replace('\n', ' ').strip()
        
        markdown_output += f"**{clean_title}**\n\n*By: {clean_author}*\n\n---\n\n"
        
    # 3. Return the string (Gradio's gr.Markdown will successfully process this)
    return markdown_output

demo = gr.Interface(
    fn=agent_chat,
    inputs=gr.Textbox(lines=2, placeholder="E.g., Find 2 recent papers on Agentic AI and format them as BibTeX."),
    outputs=gr.Markdown(label="Agent Output"),
    title="Autonomous arXiv Research Agent",
    description="Powered by smolagents and Qwen 2.5 Coder"
)

demo.launch()
