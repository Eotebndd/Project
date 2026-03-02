# 📊 Data Explorer Agent

基于 **LangChain** 框架的智能数据分析助手，支持自然语言交互完成数据探索任务。

## ✨ 功能特性

### 核心功能

| 功能 | 描述 | 示例问题 |
|------|------|----------|
| 📊 数据摘要统计 | 自动计算均值、方差、最大最小值等统计量 | "给我数据的统计摘要" |
| 📈 数据可视化 | 绘制分布图、柱状图、饼图、相关性热力图等 | "画出Age列的分布图" |
| 🤖 模型训练预测 | 使用sklearn训练分类/回归模型 | "训练模型预测Survived列" |
| 🔗 相关性分析 | 分析数值列之间的相关性 | "分析数据的相关性" |

### 界面特性

- **ChatGPT风格界面**：简洁现代的对话式交互
- **文件上传**：支持拖拽上传任意CSV文件
- **思考过程展示**：可折叠的Agent思考过程
- **图表展示**：生成的图表直接嵌入对话中
- **快捷问题**：一键选择常用分析任务

### 技术特点

- **通用数据支持**：适用于任何CSV数据集，不绑定特定文件
- **自动任务识别**：自动判断分类/回归任务
- **会话管理**：支持多会话、新对话
- **多模型支持**：OpenAI、DeepSeek等兼容API

## 📁 项目结构

```
Project/
├── config.py           # 配置管理（python-dotenv）
├── tools.py            # LangChain自定义工具
│   ├── DataSummaryTool      # 数据摘要
│   ├── DataVisualizationTool # 数据可视化
│   └── ModelTrainingTool    # 模型训练
├── main.py             # FastAPI后端服务
├── titanic_agent.py    # 命令行版本
├── test_tools.py       # 工具测试脚本
├── requirements.txt    # 项目依赖
├── .env.example        # 环境变量示例
├── .env                # 环境变量配置
├── static/
│   └── index.html      # Web前端界面
├── uploads/            # 上传文件目录
└── output_plots/       # 图表输出目录
```

## 🛠️ 安装配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件
```

`.env` 配置示例：

```bash
# DeepSeek（推荐）
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat

# 或 OpenAI
OPENAI_API_KEY=sk-xxxxx
MODEL_NAME=gpt-4o-mini
```

## 🚀 运行方式

### Web服务（推荐）

```bash
python main.py
```

访问 http://localhost:8000

### 命令行版本

```bash
# 交互模式
python titanic_agent.py

# 演示模式
python titanic_agent.py --demo
```

### 测试工具（无需API Key）

```bash
python test_tools.py
```

## 📖 使用指南

### Web界面使用

1. **上传数据**：点击或拖拽上传CSV文件
2. **选择快捷问题**或**输入自定义问题**
3. **查看结果**：文字结果、思考过程、图表都在对话中展示
4. **继续对话**：可以继续追问或要求更多分析

### 示例对话

```
用户: 给我数据的统计摘要

Agent: === 数据统计摘要 ===

数据集形状: 891 行, 12 列

列名: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', ...]

【数值型列统计】
       PassengerId    Survived      Pclass         Age
count   891.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118
...
```

```
用户: 画出Survived列的分布图

Agent: 已成功绘制 'Survived' 列的分布图！
[图表展示在对话中]
```

```
用户: 训练模型预测Survived

Agent: === 模型训练结果 ===

任务类型: 分类任务
模型: Random Forest Classifier
目标列: Survived
准确率: 79.89%

[特征重要性图表和混淆矩阵展示在对话中]
```

## 🔧 API接口

### POST /api/upload
上传CSV文件

### POST /api/chat
发送消息与Agent对话

```json
{
  "message": "给我数据摘要",
  "session_id": "xxx"
}
```

### GET /api/session/{session_id}
获取会话信息

## 🧩 扩展开发

### 添加新工具

在 `tools.py` 中添加：

```python
class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "工具描述"
    args_schema: Type[BaseModel] = MyToolInput
    
    def _run(self, query: str, session_id: str = None) -> str:
        df = get_df(session_id)
        # 实现逻辑
        return "结果"
```

## 📝 技术栈

- **框架**: LangChain
- **后端**: FastAPI + Uvicorn
- **前端**: HTML/CSS/JavaScript（原生）
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib
- **机器学习**: Scikit-learn
- **LLM**: OpenAI API / DeepSeek API

## 📄 License

MIT License
