# 🚢 泰坦尼克号数据探索Agent

基于 **LangChain** 框架的智能数据分析助手，支持自然语言交互完成数据探索任务。

## ✨ 功能特性

### 三大核心功能

| 功能 | 描述 | 示例问题 |
|------|------|----------|
| 📊 数据摘要统计 | 自动计算均值、方差、最大最小值等统计量 | "请给我数据的统计摘要" |
| 📈 数据可视化 | 绘制分布图、柱状图、饼图等 | "画出Survived列的分布" |
| 🤖 模型训练预测 | 使用sklearn训练模型并预测生存率 | "训练一个模型预测Survived" |

### 技术特点

- **自然语言交互**：通过对话方式完成数据分析，无需编写代码
- **LLM驱动**：所有功能均由大语言模型调用工具完成
- **多模型支持**：支持OpenAI、DeepSeek等兼容API
- **Web界面**：提供友好的前端界面，支持快捷操作和对话交互

## 📁 项目结构

```
Project/
├── config.py           # 配置管理（使用python-dotenv）
├── tools.py            # 自定义LangChain工具
│   ├── DataSummaryTool      # 数据摘要工具
│   ├── DataVisualizationTool # 数据可视化工具
│   └── ModelTrainingTool    # 模型训练工具
├── main.py             # FastAPI后端服务
├── titanic_agent.py    # 命令行版本Agent
├── test_tools.py       # 工具测试脚本（无需API Key）
├── requirements.txt    # 项目依赖
├── .env.example        # 环境变量示例
├── .env                # 环境变量配置（需自行创建）
├── static/
│   └── index.html      # Web前端界面
├── output_plots/       # 生成的图表输出目录
└── titanic_cleaned.csv # 泰坦尼克号数据集
```

## 🛠️ 安装步骤

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

`.env` 文件配置说明：

```bash
# API密钥（必填）
OPENAI_API_KEY=your-api-key-here

# API基础URL（可选，用于DeepSeek等其他服务）
OPENAI_BASE_URL=https://api.deepseek.com/v1

# 模型名称
MODEL_NAME=deepseek-chat
```

#### 支持的模型配置

**DeepSeek（推荐，性价比高）**
```bash
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
```

**OpenAI**
```bash
OPENAI_API_KEY=sk-xxxxx
MODEL_NAME=gpt-4o-mini
```

## 🚀 运行方式

### 方式一：Web服务（推荐）

```bash
python main.py
```

访问 http://localhost:8000 即可使用Web界面。

**功能说明：**
- 点击快捷按钮执行常用功能
- 在对话框输入问题与Agent交互
- 自动显示生成的图表

### 方式二：命令行交互

```bash
python titanic_agent.py
```

进入交互模式，输入问题与Agent对话。

### 方式三：演示模式

```bash
python titanic_agent.py --demo
```

自动执行三个核心功能演示。

### 方式四：测试工具（无需API Key）

```bash
python test_tools.py
```

直接测试三个工具功能，验证代码正确性，无需配置API Key。

## 📖 使用示例

### 数据摘要统计

```
用户: 请给我数据的统计摘要

Agent: === 数据统计摘要 ===

数据集形状: 891 行, 12 列

数值型列的统计信息:
       PassengerId    Survived      Pclass         Age       SibSp  \
count   891.000000  891.000000  891.000000  891.000000  891.000000   
mean    446.000000    0.383838    2.308642   29.699118    0.523008   
...
```

### 数据可视化

```
用户: 画出Survived列的分布

Agent: 已成功绘制Survived列的分布图！
图表已保存至: output_plots/survived_distribution.png

统计信息:
- 未生存(0): 549人 (61.6%)
- 生存(1): 342人 (38.4%)
```

### 模型训练

```
用户: 训练一个模型预测Survived

Agent: === 模型训练结果 ===

使用模型: Random Forest Classifier
特征列: Pclass, Age, SibSp, Parch, Fare
训练集大小: 712
测试集大小: 179

=== 模型性能 ===
准确率 (Accuracy): 0.7989 (79.89%)

=== 特征重要性 ===
  Fare: 0.2845
  Age: 0.2512
  Pclass: 0.1823
  ...
```

## 🔧 API接口

### POST /api/chat

与Agent对话

```json
{
  "message": "请给我数据的统计摘要",
  "session_id": "optional-session-id"
}
```

响应：
```json
{
  "response": "Agent的回答内容",
  "session_id": "session-id",
  "image_paths": ["/outputs/survived_distribution.png"]
}
```

### POST /api/summary

获取数据摘要

### POST /api/visualize

生成可视化图表

### POST /api/train

训练模型

## 🧩 扩展开发

### 添加新工具

在 `tools.py` 中添加新的工具类：

```python
class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "工具描述"
    args_schema: Type[BaseModel] = MyToolInput
    
    def _run(self, query: str) -> str:
        # 实现工具逻辑
        return "结果"
```

然后在 `get_tools()` 函数中注册：

```python
def get_tools():
    return [
        DataSummaryTool(),
        DataVisualizationTool(),
        ModelTrainingTool(),
        MyCustomTool()  # 添加新工具
    ]
```

## 📊 数据集说明

使用泰坦尼克号乘客数据集（titanic_cleaned.csv），包含891条记录，12个字段：

| 字段 | 说明 |
|------|------|
| PassengerId | 乘客ID |
| Survived | 是否生存（0=否，1=是）- **目标变量** |
| Pclass | 船舱等级（1/2/3） |
| Name | 姓名 |
| Sex | 性别 |
| Age | 年龄 |
| SibSp | 船上兄弟姐妹/配偶数量 |
| Parch | 船上父母/子女数量 |
| Ticket | 票号 |
| Fare | 票价 |
| Cabin | 船舱号 |
| Embarked | 登船港口 |

## 📝 技术栈

- **框架**: LangChain
- **后端**: FastAPI
- **前端**: HTML/CSS/JavaScript
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib
- **机器学习**: Scikit-learn
- **LLM**: OpenAI API / DeepSeek API

## 📄 License

MIT License
