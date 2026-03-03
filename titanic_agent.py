"""
泰坦尼克号数据探索Agent - 命令行版本
基于LangChain框架，支持自然语言交互
"""
import os
import sys
import config
from tools import get_tools
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate


def create_agent():
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        print("错误: 请先配置API Key")
        print("1. 复制 .env.example 为 .env")
        print("2. 编辑 .env 文件，填入你的API Key")
        return None
    
    llm_kwargs = {"model": config.MODEL_NAME, "temperature": 0}
    if config.OPENAI_BASE_URL:
        llm_kwargs["base_url"] = config.OPENAI_BASE_URL
    
    llm = ChatOpenAI(**llm_kwargs)
    tools = get_tools()
    
    prompt = PromptTemplate.from_template(
        """你是一个专业的数据分析助手。

你可以使用以下工具:

{tools}

工具名称: {tool_names}

使用工具时，请严格遵循以下格式（每行必须独立，不能跨行）:

Question: 用户的问题
Thought: 简短思考要做什么（必须在一行内完成，不要换行）
Action: 要使用的工具名称，必须是 [{tool_names}] 中的一个
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (这个 Thought/Action/Action Input/Observation 可以重复N次)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

重要提示: Thought 必须是单行，不能包含换行符

开始!

Question: {input}
Thought: {agent_scratchpad}"""
    )
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)


def run_interactive(agent_executor):
    print("\n" + "="*60)
    print("泰坦尼克号数据探索Agent - 基于LangChain")
    print("="*60)
    print("\n可用功能:")
    print("1. 数据摘要统计 - 输入如: '请给我数据的统计摘要'")
    print("2. 数据可视化 - 输入如: '画出Survived列的分布'")
    print("3. 模型训练 - 输入如: '训练一个模型预测Survived'")
    print("4. 输入 'quit' 或 'exit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用，再见！")
                break
            
            if not user_input:
                continue
            
            print("\n" + "-"*40)
            print("Agent正在处理...")
            print("-"*40 + "\n")
            
            result = agent_executor.invoke({"input": user_input})
            
            print("\n" + "="*40)
            print("结果:")
            print("="*40)
            print(result["output"])
            
        except KeyboardInterrupt:
            print("\n\n程序已中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")


if __name__ == "__main__":
    print(f"使用模型: {config.MODEL_NAME}")
    
    agent_executor = create_agent()
    if not agent_executor:
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_features(agent_executor)
    else:
        run_interactive(agent_executor)
