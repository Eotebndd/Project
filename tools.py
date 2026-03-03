"""
自定义工具 - 数据摘要、可视化、模型训练
支持动态数据集，不绑定特定CSV文件
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from langchain.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
import warnings
import config

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_session_data: Dict[str, Dict[str, Any]] = {}

def set_session_data(session_id: str, df: pd.DataFrame, file_path: str):
    global _session_data
    _session_data[session_id] = {
        'df': df,
        'file_path': file_path,
        'output_dir': os.path.join(config.OUTPUT_DIR, session_id),
        'image_paths': []
    }
    os.makedirs(_session_data[session_id]['output_dir'], exist_ok=True)

def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    return _session_data.get(session_id)

def get_df(session_id: str) -> Optional[pd.DataFrame]:
    data = get_session_data(session_id)
    if data:
        return data['df']
    return None

def get_output_dir(session_id: str) -> str:
    data = get_session_data(session_id)
    if data:
        return data['output_dir']
    return config.OUTPUT_DIR

def add_image_path(session_id: str, path: str):
    global _session_data
    if session_id in _session_data:
        _session_data[session_id]['image_paths'].append(path)

def get_and_clear_image_paths(session_id: str) -> list:
    global _session_data
    if session_id in _session_data:
        paths = _session_data[session_id]['image_paths'].copy()
        _session_data[session_id]['image_paths'] = []
        return paths
    return []


class DataSummaryInput(BaseModel):
    query: str = Field(default="", description="用户关于数据摘要的问题")


class DataSummaryTool(BaseTool):
    name: str = "data_summary"
    description: str = "获取数据的统计摘要信息，包括均值、方差、最大值、最小值、计数等统计量。适用于任何CSV数据集。输入参数query为用户的问题。"
    args_schema: Type[BaseModel] = DataSummaryInput
    
    def _run(self, query: str = "") -> str:
        session_id = getattr(self, '_session_id', None)
        
        df = get_df(session_id)
        if df is None:
            return "错误: 未找到数据，请先上传CSV文件"
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        result = "## 数据统计摘要\n\n"
        result += f"**数据集形状**: {df.shape[0]} 行, {df.shape[1]} 列\n\n"
        result += f"**列名**: {', '.join(df.columns.tolist())}\n\n"
        
        if numeric_cols:
            result += "### 数值型列统计\n\n"
            summary_stats = df[numeric_cols].describe()
            
            stat_rows = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            cols = list(summary_stats.columns)
            
            header = "| 统计量 | " + " | ".join(cols) + " |"
            separator = "|--------|" + "|".join(["--------" for _ in cols]) + "|"
            
            result += header + "\n"
            result += separator + "\n"
            
            for stat in stat_rows:
                if stat in summary_stats.index:
                    values = [f"{summary_stats.loc[stat, col]:.2f}" for col in cols]
                    row = f"| {stat} | " + " | ".join(values) + " |"
                    result += row + "\n"
            result += "\n"
        
        if cat_cols:
            result += "### 分类型列统计\n\n"
            result += "| 列名 | 唯一值数量 |\n|------|------------|\n"
            for col in cat_cols[:5]:
                unique_count = df[col].nunique()
                result += f"| {col} | {unique_count} |\n"
            if len(cat_cols) > 5:
                result += f"| ... | 还有 {len(cat_cols) - 5} 列 |\n"
            result += "\n"
        
        result += "### 缺失值统计\n\n"
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]
        if len(missing_cols) > 0:
            result += "| 列名 | 缺失数量 | 缺失比例 |\n|------|----------|----------|\n"
            for col, count in missing_cols.items():
                result += f"| {col} | {count} | {count/len(df)*100:.1f}% |\n"
        else:
            result += "无缺失值\n"
        
        return result
    
    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class DataVisualizationInput(BaseModel):
    query: str = Field(default="", description="用户关于数据可视化的问题，例如'画出Age列的分布图'")


class DataVisualizationTool(BaseTool):
    name: str = "data_visualization"
    description: str = "对数据进行可视化绘图。可以绘制分布图、柱状图、饼图、相关性热力图等。支持同时绘制多个图表。输入参数query中应包含要绘制的列名或图表类型。"
    args_schema: Type[BaseModel] = DataVisualizationInput
    
    def _run(self, query: str = "") -> str:
        session_id = getattr(self, '_session_id', None)
        
        df = get_df(session_id)
        if df is None:
            return "错误: 未找到数据，请先上传CSV文件"
        
        output_dir = get_output_dir(session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        query_lower = query.lower()
        
        try:
            columns = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 检测相关性热力图请求
            if "correlation" in query_lower or "相关" in query_lower or "热力图" in query_lower:
                return self._plot_correlation(df, output_dir, session_id)
            
            # 检测散点图请求
            if "scatter" in query_lower or "散点" in query_lower:
                return self._plot_scatter(df, query_lower, columns, numeric_cols, output_dir, session_id)
            
            # 检测折线图请求
            if "line" in query_lower or "折线" in query_lower or "趋势" in query_lower:
                return self._plot_line(df, query_lower, columns, numeric_cols, output_dir, session_id)
            
            # 检测饼图请求
            if "pie" in query_lower or "饼图" in query_lower:
                return self._plot_pie(df, query_lower, columns, cat_cols, output_dir, session_id)
            
            # 智能匹配列名
            mentioned_cols = []
            for col in columns:
                col_lower = col.lower()
                # 精确匹配或包含匹配
                if col_lower in query_lower or col in query:
                    mentioned_cols.append(col)
                # 处理列名中的空格和特殊字符
                elif any(word in query_lower for word in col_lower.replace('_', ' ').split()):
                    mentioned_cols.append(col)
            
            # 去重
            mentioned_cols = list(dict.fromkeys(mentioned_cols))
            
            if len(mentioned_cols) > 1:
                return self._plot_multiple_columns(df, mentioned_cols, output_dir, session_id, query_lower)
            
            if mentioned_cols:
                return self._plot_column(df, mentioned_cols[0], output_dir, session_id, query_lower)
            
            return self._plot_overview(df, output_dir, session_id)
            
        except Exception as e:
            return f"绘图时发生错误: {str(e)}"
    
    def _plot_multiple_columns(self, df, cols, output_dir, session_id, query_lower=""):
        results = []
        
        # 检测是否需要绘制对比图
        if "对比" in query_lower or "compare" in query_lower or len(cols) == 2:
            return self._plot_comparison(df, cols, output_dir, session_id)
        
        for col in cols:
            result = self._plot_column(df, col, output_dir, session_id, query_lower)
            results.append(result)
        return "\n\n".join(results)
    
    def _plot_comparison(self, df, cols, output_dir, session_id):
        """绘制两列对比图"""
        if len(cols) < 2:
            return "需要至少两列进行对比"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        col1, col2 = cols[0], cols[1]
        
        # 散点图
        if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
            axes[0].scatter(df[col1], df[col2], alpha=0.5, color='#45b7d1')
            axes[0].set_xlabel(col1)
            axes[0].set_ylabel(col2)
            axes[0].set_title(f'{col1} vs {col2}')
            
            # 分布对比
            axes[1].hist(df[col1].dropna(), bins=30, alpha=0.5, label=col1, color='#45b7d1')
            axes[1].hist(df[col2].dropna(), bins=30, alpha=0.5, label=col2, color='#ff6b6b')
            axes[1].legend()
            axes[1].set_title('Distribution Comparison')
        else:
            # 分类数据对比
            for i, col in enumerate([col1, col2]):
                value_counts = df[col].value_counts().head(10)
                axes[i].barh(range(len(value_counts)), value_counts.values, color='#4ecdc4')
                axes[i].set_yticks(range(len(value_counts)))
                axes[i].set_yticklabels(value_counts.index.astype(str))
                axes[i].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{col1}_{col2}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制 '{col1}' 和 '{col2}' 的对比图！"
    
    def _plot_scatter(self, df, query_lower, columns, numeric_cols, output_dir, session_id):
        """绘制散点图"""
        if len(numeric_cols) < 2:
            return "数值列不足，无法绘制散点图"
        
        # 尝试识别x和y列
        x_col, y_col = None, None
        for col in numeric_cols:
            if f'x={col.lower()}' in query_lower or f'x是{col.lower()}' in query_lower:
                x_col = col
            if f'y={col.lower()}' in query_lower or f'y是{col.lower()}' in query_lower:
                y_col = col
        
        if not x_col or not y_col:
            # 使用前两个数值列
            x_col, y_col = numeric_cols[0], numeric_cols[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[x_col], df[y_col], alpha=0.5, color='#45b7d1')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"scatter_{x_col}_{y_col}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制 '{x_col}' vs '{y_col}' 的散点图！"
    
    def _plot_line(self, df, query_lower, columns, numeric_cols, output_dir, session_id):
        """绘制折线图"""
        if not numeric_cols:
            return "无数值列可绘制折线图"
        
        mentioned_cols = [col for col in numeric_cols if col.lower() in query_lower]
        if not mentioned_cols:
            mentioned_cols = numeric_cols[:3]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in mentioned_cols:
            ax.plot(df[col].values, label=col, alpha=0.7)
        
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.set_title('Line Chart')
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "line_chart.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制折线图！"
    
    def _plot_pie(self, df, query_lower, columns, cat_cols, output_dir, session_id):
        """绘制饼图"""
        target_col = None
        for col in columns:
            if col.lower() in query_lower:
                target_col = col
                break
        
        if not target_col:
            if cat_cols:
                target_col = cat_cols[0]
            else:
                target_col = columns[0]
        
        value_counts = df[target_col].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(value_counts.values, labels=value_counts.index.astype(str),
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{target_col} Pie Chart')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{target_col}_pie.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制 '{target_col}' 的饼图！"
    
    def _plot_column(self, df, col, output_dir, session_id, query_lower=""):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if df[col].dtype in ['int64', 'float64']:
            if df[col].nunique() <= 10:
                counts = df[col].value_counts().sort_index()
                axes[0].bar(counts.index.astype(str), counts.values, color='#4ecdc4')
                axes[0].set_title(f'{col} Distribution (Bar Chart)')
                axes[0].set_xlabel(col)
                axes[0].set_ylabel('Count')
            else:
                axes[0].hist(df[col].dropna(), bins=30, color='#45b7d1', edgecolor='white', alpha=0.7)
                axes[0].set_title(f'{col} Distribution (Histogram)')
                axes[0].set_xlabel(col)
                axes[0].set_ylabel('Frequency')
            
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].boxplot(df[col].dropna(), vert=True)
            axes[1].set_title(f'{col} Box Plot')
            axes[1].set_ylabel(col)
        else:
            value_counts = df[col].value_counts().head(15)
            axes[0].barh(range(len(value_counts)), value_counts.values, color='#4ecdc4')
            axes[0].set_yticks(range(len(value_counts)))
            axes[0].set_yticklabels(value_counts.index.astype(str))
            axes[0].set_title(f'{col} Distribution (Top 15)')
            axes[0].set_xlabel('Count')
            
            if len(value_counts) <= 8:
                axes[1].pie(value_counts.values, labels=value_counts.index.astype(str),
                           autopct='%1.1f%%', startangle=90)
                axes[1].set_title(f'{col} Pie Chart')
            else:
                axes[1].text(0.5, 0.5, 'Too many categories\nfor pie chart', 
                           ha='center', va='center', fontsize=14)
                axes[1].set_title(f'{col}')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{col}_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制 '{col}' 列的分布图！"
    
    def _plot_correlation(self, df, output_dir, session_id):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return "数值列不足，无法绘制相关性热力图"
        
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax)
        ax.set_title('Correlation Heatmap')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制相关性热力图！"
    
    def _plot_overview(self, df, output_dir, session_id):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        
        if not numeric_cols:
            return "无数值列可绘制"
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:4]):
            if df[col].nunique() <= 10:
                counts = df[col].value_counts().sort_index()
                axes[i].bar(counts.index.astype(str), counts.values, color='#4ecdc4')
            else:
                axes[i].hist(df[col].dropna(), bins=30, color='#45b7d1', edgecolor='white', alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
        
        for i in range(len(numeric_cols), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "data_overview.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        add_image_path(session_id, save_path)
        return f"已成功绘制数据概览图！"
    
    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class ModelTrainingInput(BaseModel):
    query: str = Field(default="", description="用户关于模型训练和预测的问题，需要指定目标列名，例如'训练模型预测Survived列'")


class ModelTrainingTool(BaseTool):
    name: str = "model_training"
    description: str = "使用sklearn训练机器学习模型进行预测。输入参数query中需要指定目标列名（target column）。自动处理分类和回归任务。"
    args_schema: Type[BaseModel] = ModelTrainingInput
    
    def _run(self, query: str = "") -> str:
        session_id = getattr(self, '_session_id', None)
        
        df = get_df(session_id)
        if df is None:
            return "错误: 未找到数据，请先上传CSV文件"
        
        output_dir = get_output_dir(session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            target_col = self._extract_target_column(query, df)
            if not target_col:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    target_col = numeric_cols[-1]
                    result = f"未指定目标列，自动选择 '{target_col}' 作为目标变量\n\n"
                else:
                    return "无法确定目标列，请在问题中指定要预测的列名"
            else:
                result = f"目标列: '{target_col}'\n\n"
            
            return result + self._train_model(df, target_col, output_dir)
            
        except Exception as e:
            return f"模型训练时发生错误: {str(e)}"
    
    def _extract_target_column(self, query, df):
        columns = df.columns.tolist()
        query_lower = query.lower()
        
        keywords = ['预测', 'predict', 'target', '目标', 'y是', 'y is']
        for keyword in keywords:
            if keyword in query_lower:
                for col in columns:
                    if col.lower() in query_lower:
                        return col
        
        for col in columns:
            if col.lower() in query_lower:
                return col
        
        return None
    
    def _train_model(self, df, target_col, output_dir):
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        if not feature_cols:
            return "没有可用的特征列进行训练"
        
        df_model = df[feature_cols + [target_col]].copy()
        
        for col in feature_cols:
            df_model[col] = df_model[col].fillna(df_model[col].median())
        df_model = df_model.dropna(subset=[target_col])
        
        X = df_model[feature_cols]
        y = df_model[target_col]
        
        unique_values = y.nunique()
        is_classification = unique_values <= 10 or y.dtype == 'object'
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            is_classification = True
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            result = "=== 模型训练结果 ===\n\n"
            result += f"任务类型: 分类任务\n"
            result += f"模型: Random Forest Classifier\n"
            result += f"目标列: {target_col}\n"
            result += f"特征列: {', '.join(feature_cols)}\n"
            result += f"训练集大小: {len(X_train)}\n"
            result += f"测试集大小: {len(X_test)}\n\n"
            result += f"=== 模型性能 ===\n"
            result += f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n"
            
            if unique_values <= 5:
                report = classification_report(y_test, y_pred, output_dict=True)
                result += "=== 分类报告 ===\n"
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        result += f"类别 {label}:\n"
                        result += f"  Precision: {metrics.get('precision', 0):.4f}\n"
                        result += f"  Recall: {metrics.get('recall', 0):.4f}\n"
                        result += f"  F1-score: {metrics.get('f1-score', 0):.4f}\n"
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            result = "=== 模型训练结果 ===\n\n"
            result += f"任务类型: 回归任务\n"
            result += f"模型: Random Forest Regressor\n"
            result += f"目标列: {target_col}\n"
            result += f"特征列: {', '.join(feature_cols)}\n"
            result += f"训练集大小: {len(X_train)}\n"
            result += f"测试集大小: {len(X_test)}\n\n"
            result += f"=== 模型性能 ===\n"
            result += f"均方误差 (MSE): {mse:.4f}\n"
            result += f"R² 分数: {r2:.4f}\n\n"
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result += "=== 特征重要性 ===\n"
        for _, row in feature_importance.iterrows():
            result += f"  {row['feature']}: {row['importance']:.4f}\n"
        
        save_path = self._plot_model_results(model, X_test, y_test, y_pred, feature_importance, 
                                 is_classification, output_dir)
        session_id = getattr(self, '_session_id', None)
        add_image_path(session_id, save_path)
        
        return result
    
    def _plot_model_results(self, model, X_test, y_test, y_pred, feature_importance, 
                           is_classification, output_dir):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        feature_importance.plot(x='feature', y='importance', kind='bar', ax=axes[0], color='#45b7d1')
        axes[0].set_title('Feature Importance')
        axes[0].set_xlabel('Feature')
        axes[0].set_ylabel('Importance')
        axes[0].legend().remove()
        axes[0].tick_params(axis='x', rotation=45)
        
        if is_classification:
            cm = confusion_matrix(y_test, y_pred)
            im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1].set_title('Confusion Matrix')
            n_classes = cm.shape[0]
            axes[1].set_xticks(range(n_classes))
            axes[1].set_yticks(range(n_classes))
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            
            for i in range(n_classes):
                for j in range(n_classes):
                    axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                               color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)
        else:
            axes[1].scatter(y_test, y_pred, alpha=0.5, color='#45b7d1')
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[1].set_title('Actual vs Predicted')
            axes[1].set_xlabel('Actual')
            axes[1].set_ylabel('Predicted')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "model_results.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    
    async def _arun(self, query: str = "") -> str:
        return self._run(query)


def get_tools(session_id: str = None):
    tools = [
        DataSummaryTool(),
        DataVisualizationTool(),
        ModelTrainingTool()
    ]
    for tool in tools:
        tool._session_id = session_id
    return tools
