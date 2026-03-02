"""
FastAPI后端服务 - 泰坦尼克号数据探索Agent
"""
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

import config
from tools import get_tools

app = FastAPI(title="Titanic Data Exploration Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")

agent_cache = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    image_paths: list[str] = []


def create_agent_executor():
    llm_kwargs = {"model": config.MODEL_NAME, "temperature": 0}
    if config.OPENAI_BASE_URL:
        llm_kwargs["base_url"] = config.OPENAI_BASE_URL
    
    llm = ChatOpenAI(**llm_kwargs)
    tools = get_tools()
    
    prompt = PromptTemplate.from_template(
        """你是一个专业的数据分析助手，帮助用户分析泰坦尼克号数据集。

你可以使用以下工具:

{tools}

工具名称: {tool_names}

使用工具时，请遵循以下格式:

Question: 用户的问题
Thought: 你应该思考要做什么
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

开始!

Question: {input}
Thought: {agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)


def extract_image_paths(text: str) -> list[str]:
    import re
    pattern = r'图表已保存至: (.+?\.png)'
    matches = re.findall(pattern, text)
    paths = []
    for match in matches:
        filename = os.path.basename(match)
        paths.append(f"/outputs/{filename}")
    return paths


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Please create static/index.html</h1>", status_code=200)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        raise HTTPException(status_code=500, detail="API Key未配置，请检查.env文件")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    if session_id not in agent_cache:
        agent_cache[session_id] = create_agent_executor()
    
    agent_executor = agent_cache[session_id]
    
    try:
        result = agent_executor.invoke({"input": request.message})
        response_text = result["output"]
        image_paths = extract_image_paths(response_text)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            image_paths=image_paths
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent处理错误: {str(e)}")


@app.post("/api/summary")
async def get_summary():
    from tools import DataSummaryTool
    tool = DataSummaryTool()
    result = tool._run("获取数据摘要")
    return {"result": result}


@app.post("/api/visualize")
async def visualize(column: str = "survived"):
    from tools import DataVisualizationTool
    tool = DataVisualizationTool()
    result = tool._run(f"画出{column}列的分布")
    image_paths = extract_image_paths(result)
    return {"result": result, "image_paths": image_paths}


@app.post("/api/train")
async def train_model():
    from tools import ModelTrainingTool
    tool = ModelTrainingTool()
    result = tool._run("训练模型预测Survived")
    image_paths = extract_image_paths(result)
    return {"result": result, "image_paths": image_paths}


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(config.OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
