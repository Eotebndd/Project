"""
FastAPI后端服务 - 数据探索Agent
支持文件上传、会话管理、流式输出
"""
import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.prompts import PromptTemplate
from langchain_classic.callbacks.base import BaseCallbackHandler

import config
from tools import get_tools, set_session_data, get_session_data, get_and_clear_image_paths

CHECKPOINT_DIR = "checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PROMPT_TEMPLATE = """你是一个专业的数据分析助手，帮助用户分析CSV数据集。

你可以使用以下工具:

{tools}

工具名称: {tool_names}

使用工具时，请严格遵循以下格式:

Question: 用户的问题
Thought: 简短思考要做什么
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

重要提示:
1. Thought 必须是单行，不能包含换行符
2. 如果用户询问数据摘要，使用data_summary工具
3. 如果用户要求画图或可视化，使用data_visualization工具
4. 如果用户要求训练模型或预测，使用model_training工具，需要指定目标列名
5. Final Answer 必须包含工具返回的完整内容，包括表格、统计数据等，不要省略

开始!

Question: {input}
Thought: {agent_scratchpad}"""


def save_conversation_log(session_id: str, user_message: str, agent_response: str, 
                          image_paths: list, thinking: str = "", tool_calls: list = None):
    filename = f"session_{session_id[:8]}.log"
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    is_new_file = not os.path.exists(filepath)
    
    log_content = ""
    if is_new_file:
        log_content += f"# 会话日志\n\n"
        log_content += f"**会话ID**: `{session_id}`\n\n"
        log_content += f"**创建时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        log_content += "---\n"
    
    log_content += f"\n## 对话 [{datetime.now().strftime('%H:%M:%S')}]\n\n"
    log_content += f"**用户**: {user_message}\n\n"
    
    if thinking:
        log_content += f"<details>\n<summary>思考过程</summary>\n\n```\n{thinking}\n```\n</details>\n\n"
    
    if tool_calls:
        log_content += f"<details>\n<summary>工具调用 ({len(tool_calls)}次)</summary>\n\n"
        for i, call in enumerate(tool_calls, 1):
            log_content += f"```\n{i}. {call}\n```\n\n"
        log_content += "</details>\n\n"
    
    log_content += f"**Agent**: {agent_response}\n\n"
    
    if image_paths:
        log_content += f"**生成的图片**:\n"
        for path in image_paths:
            log_content += f"- `{path}`\n"
        log_content += "\n"
    
    log_content += "---\n"
    
    mode = 'w' if is_new_file else 'a'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(log_content)
    
    return filepath


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


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    image_paths: list[str] = []
    thinking: str = ""


class SimpleCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.thoughts = []
        self.tool_calls = []
    
    def on_agent_action(self, action, **kwargs):
        thought = action.log if hasattr(action, 'log') else str(action)
        self.thoughts.append(thought)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.tool_calls.append(f"调用工具: {tool_name}\n  输入: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        short_output = output[:300] + "..." if len(output) > 300 else output
        if self.tool_calls:
            self.tool_calls[-1] += f"\n  输出: {short_output}"
    
    def get_thinking_process(self):
        return "\n".join(self.thoughts)
    
    def get_tool_calls(self):
        return self.tool_calls


class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.thoughts = []
        self.tool_calls = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.queue.put_nowait(json.dumps({"type": "thinking", "content": "正在分析问题..."}))
    
    def on_agent_action(self, action, **kwargs):
        thought = action.log if hasattr(action, 'log') else str(action)
        self.thoughts.append(thought)
        self.queue.put_nowait(json.dumps({"type": "thinking", "content": f"思考中...\n{thought}"}))
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        self.tool_calls.append(f"调用工具: {tool_name}\n  输入: {input_str}")
        self.queue.put_nowait(json.dumps({"type": "thinking", "content": f"正在调用工具: {tool_name}"}))
    
    def on_tool_end(self, output, **kwargs):
        short_output = output[:300] + "..." if len(output) > 300 else output
        if self.tool_calls:
            self.tool_calls[-1] += f"\n  输出: {short_output}"
        self.queue.put_nowait(json.dumps({"type": "thinking", "content": f"工具执行完成:\n{short_output}"}))
    
    def get_thinking_process(self):
        return "\n".join(self.thoughts)
    
    def get_tool_calls(self):
        return self.tool_calls


def create_agent_executor(session_id: str, callback=None):
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
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, 
        max_iterations=10
    )


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
    
    if not get_session_data(session_id):
        raise HTTPException(status_code=400, detail="会话已过期，请重新上传数据")
    
    callback = SimpleCallbackHandler()
    agent_executor = create_agent_executor(session_id, callback)
    
    try:
        result = agent_executor.invoke({"input": request.message})
        response_text = result["output"]
        
        raw_paths = get_and_clear_image_paths(session_id)
        image_paths = [f"/outputs/{session_id}/{os.path.basename(p)}" for p in raw_paths]
        
        save_conversation_log(
            session_id, request.message, response_text, image_paths,
            callback.get_thinking_process(), callback.get_tool_calls()
        )
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            image_paths=image_paths,
            thinking=callback.get_thinking_process()
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
    
    if not get_session_data(session_id):
        raise HTTPException(status_code=400, detail="会话已过期")
    
    queue = asyncio.Queue()
    finished = False
    
    def run_agent_sync():
        nonlocal finished
        callback = StreamCallbackHandler(queue)
        agent_executor = create_agent_executor(session_id, callback)
        
        try:
            result = agent_executor.invoke({"input": request.message})
            response_text = result["output"]
            
            raw_paths = get_and_clear_image_paths(session_id)
            image_paths = [f"/outputs/{session_id}/{os.path.basename(p)}" for p in raw_paths]
            
            save_conversation_log(
                session_id, request.message, response_text, image_paths,
                callback.get_thinking_process(), callback.get_tool_calls()
            )
            
            queue.put_nowait(json.dumps({
                "type": "done", 
                "content": response_text,
                "image_paths": image_paths
            }))
        except Exception as e:
            queue.put_nowait(json.dumps({"type": "error", "content": str(e)}))
        finally:
            finished = True
    
    async def generate():
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, run_agent_sync)
        
        try:
            while not finished:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield f"data: {data}\n\n"
                    
                    if json.loads(data).get("type") in ["done", "error"]:
                        break
                except asyncio.TimeoutError:
                    continue
            
            await task
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
