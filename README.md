# Data Exploration Agent

基于 LangChain 的智能数据分析助手，支持自然语言交互进行数据探索、可视化和模型训练。

## 功能特性

- **数据摘要**: 自动生成数据集的统计摘要，包括数值型列统计、分类型列统计和缺失值分析
- **数据可视化**: 支持多种图表类型（分布图、柱状图、饼图、散点图、相关性热力图等）
- **模型训练**: 自动训练机器学习模型进行预测，支持分类和回归任务
- **流式输出**: 支持 SSE 流式响应，实时展示 Agent 思考过程
- **对话日志**: 自动保存对话记录到 checkpoint 文件夹

## 项目结构

```
Project/
├── main.py              # FastAPI 后端服务
├── tools.py             # LangChain 工具定义
├── config.py            # 配置文件
├── titanic_agent.py     # 命令行版本
├── static/
│   └── index.html       # 前端界面
├── checkpoint/          # 对话日志存储
├── uploads/             # 上传文件存储
├── output_plots/        # 生成的图表存储
├── .env                 # 环境变量配置
├── .env.example         # 环境变量示例
└── requirements.txt     # 依赖列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于自定义 API 地址
MODEL_NAME=gpt-4o-mini                       # 模型名称
```

### 3. 启动服务

**Web 版本**：
```bash
python main.py
```
访问 http://localhost:8004

**命令行版本**：
```bash
python titanic_agent.py
```

## API 接口

### 上传文件
```
POST /api/upload
Content-Type: multipart/form-data

参数: file (CSV文件)
返回: session_id, rows, columns
```

### 对话接口
```
POST /api/chat
Content-Type: application/json

参数: { "message": "问题", "session_id": "会话ID" }
返回: { "response": "回答", "image_paths": ["图片路径"] }
```

### 流式对话
```
POST /api/chat/stream
Content-Type: application/json

参数: { "message": "问题", "session_id": "会话ID" }
返回: SSE 流式数据
```

## 使用示例

### 数据摘要
```
用户: 给我数据的统计摘要
Agent: ## 数据统计摘要
       **数据集形状**: 891 行, 12 列
       ### 数值型列统计
       | 统计量 | PassengerId | Survived | ... |
       |--------|-------------|----------|-----|
       | count  | 891.00      | 891.00   | ... |
       ...
```

### 数据可视化
```
用户: 画出 Age 列的分布图
Agent: [生成 Age 列分布图并展示]
```

### 模型训练
```
用户: 训练一个模型预测 Survived
Agent: [训练模型并展示结果]
       准确率: 74.30%
       特征重要性: Fare (30.8%), Age (22.8%), ...
```

## 技术栈

- **后端**: FastAPI, LangChain, pandas, scikit-learn, matplotlib
- **前端**: HTML/CSS/JavaScript (原生)
- **AI**: OpenAI API (支持兼容接口)

## 注意事项

1. 确保 `.env` 文件中的 API Key 正确
2. 上传的文件必须是 CSV 格式
3. 生成的图表保存在 `output_plots/` 目录
4. 对话日志保存在 `checkpoint/` 目录
