"""
FastAPI后端服务 - 数据探索Agent
支持文件上传、会话管理、流式输出
"""
import os
import uuid
import json
import asyncio
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

import config
from tools import get_tools, set_session_data, get_session_data

app = FastAPI(title="Data Exploration Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
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
    thinking: str = ""


class ThinkingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thoughts = []
        self.current_thought = ""
    
    def on_agent_action(self, action, **kwargs):
        self.current_thought = f"思考: {action.log}"
        self.thoughts.append(self.current_thought)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.thoughts.append(f"调用工具: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        self.thoughts.append(f"工具输出: {output[:200]}..." if len(output) > 200 else f"工具输出: {output}")
    
    def get_thinking_process(self):
        return "\n".join(self.thoughts)


def create_agent_executor(session_id: str, callback: ThinkingCallbackHandler = None):
    llm_kwargs = {
        "model": config.MODEL_NAME,
        "temperature": 0,
    }
    if config.OPENAI_BASE_URL:
        llm_kwargs["base_url"] = config.OPENAI_BASE_URL
    
    if callback:
        llm_kwargs["callbacks"] = [callback]
    
    llm = ChatOpenAI(**llm_kwargs)
    tools = get_tools(session_id)
    
    prompt = PromptTemplate.from_template(
        """你是一个专业的数据分析助手，帮助用户分析CSV数据集。

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

重要提示:
1. 如果用户询问数据摘要，使用data_summary工具
2. 如果用户要求画图或可视化，使用data_visualization工具
3. 如果用户要求训练模型或预测，使用model_training工具，需要指定目标列名

开始!

Question: {input}
Thought: {agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, 
        max_iterations=10
    )


def extract_image_paths(text: str, session_id: str) -> list[str]:
    import re
    pattern = r'图表路径: (.+?\.png)'
    matches = re.findall(pattern, text)
    paths = []
    for match in matches:
        filename = os.path.basename(match)
        paths.append(f"/outputs/{session_id}/{filename}")
    return paths


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content="<h1>Please create static/index.html</h1>", status_code=200)


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="只支持CSV文件")
    
    session_id = str(uuid.uuid4())
    file_path = os.path.join(config.UPLOAD_DIR, f"{session_id}_{file.filename}")
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    try:
        df = pd.read_csv(file_path)
        set_session_data(session_id, df, file_path)
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": f"成功上传 {file.filename}，共 {len(df)} 行数据"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"解析CSV失败: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        raise HTTPException(status_code=500, detail="API Key未配置，请检查.env文件")
    
    session_id = request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="请先上传数据文件")
    
    session_data = get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="会话已过期，请重新上传数据")
    
    callback = ThinkingCallbackHandler()
    
    if session_id not in agent_cache:
        agent_cache[session_id] = create_agent_executor(session_id, callback)
    else:
        agent_cache[session_id] = create_agent_executor(session_id, callback)
    
    agent_executor = agent_cache[session_id]
    
    try:
        result = agent_executor.invoke({"input": request.message})
        response_text = result["output"]
        thinking_process = callback.get_thinking_process()
        image_paths = extract_image_paths(response_text, session_id)
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            image_paths=image_paths,
            thinking=thinking_process
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent处理错误: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        raise HTTPException(status_code=500, detail="API Key未配置")
    
    session_id = request.session_id
    if not session_id:
        raise HTTPException(status_code=400, detail="请先上传数据文件")
    
    session_data = get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=400, detail="会话已过期")
    
    async def generate():
        callback = ThinkingCallbackHandler()
        agent_executor = create_agent_executor(session_id, callback)
        
        yield f"data: {json.dumps({'type': 'thinking', 'content': '正在思考...'})}\n\n"
        
        try:
            result = agent_executor.invoke({"input": request.message})
            
            thinking_process = callback.get_thinking_process()
            if thinking_process:
                yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_process})}\n\n"
            
            response_text = result["output"]
            image_paths = extract_image_paths(response_text, session_id)
            
            yield f"data: {json.dumps({'type': 'response', 'content': response_text, 'image_paths': image_paths})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    session_data = get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    df = session_data['df']
    return {
        "session_id": session_id,
        "rows": len(df),
        "columns": list(df.columns),
        "file_path": session_data['file_path']
    }


@app.get("/api/images/{session_id}/{filename}")
async def get_image(session_id: str, filename: str):
    file_path = os.path.join(config.OUTPUT_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in agent_cache:
        del agent_cache[session_id]
    return {"message": "会话已删除"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
