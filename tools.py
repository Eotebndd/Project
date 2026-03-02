"""
自定义工具 - 数据摘要、可视化、模型训练
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import warnings
import config

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

_df_cache = None

def get_dataframe():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(config.DATA_PATH)
    return _df_cache

def ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


class DataSummaryInput(BaseModel):
    query: str = Field(description="用户关于数据摘要的问题")


class DataSummaryTool(BaseTool):
    name: str = "data_summary"
    description: str = "获取数据的统计摘要信息，包括均值、方差、最大值、最小值、计数等统计量。"
    args_schema: Type[BaseModel] = DataSummaryInput
    
    def _run(self, query: str) -> str:
        df = get_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary_stats = df[numeric_cols].describe()
        
        result = "=== 数据统计摘要 ===\n\n"
        result += f"数据集形状: {df.shape[0]} 行, {df.shape[1]} 列\n\n"
        result += "数值型列的统计信息:\n"
        result += summary_stats.to_string()
        result += "\n\n各列的数据类型:\n"
        result += df.dtypes.to_string()
        result += "\n\n缺失值统计:\n"
        result += df.isnull().sum().to_string()
        
        return result
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class DataVisualizationInput(BaseModel):
    query: str = Field(description="用户关于数据可视化的问题")


class DataVisualizationTool(BaseTool):
    name: str = "data_visualization"
    description: str = "对数据进行可视化绘图，例如绘制Survived列的分布图、直方图等。"
    args_schema: Type[BaseModel] = DataVisualizationInput
    
    def _run(self, query: str) -> str:
        df = get_dataframe()
        ensure_output_dir()
        query_lower = query.lower()
        
        try:
            if "survived" in query_lower or "生存" in query_lower:
                return self._plot_survived(df)
            elif "age" in query_lower or "年龄" in query_lower:
                return self._plot_age(df)
            elif "pclass" in query_lower or "舱位" in query_lower or "等级" in query_lower:
                return self._plot_pclass(df)
            elif "sex" in query_lower or "性别" in query_lower:
                return self._plot_sex(df)
            else:
                return self._plot_overall(df)
        except Exception as e:
            return f"绘图时发生错误: {str(e)}"
    
    def _plot_survived(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        survived_counts = df['Survived'].value_counts()
        
        axes[0].bar(['Not Survived (0)', 'Survived (1)'], 
                   [survived_counts.get(0, 0), survived_counts.get(1, 0)],
                   color=['#ff6b6b', '#4ecdc4'])
        axes[0].set_title('Survived Distribution (Bar Chart)')
        axes[0].set_ylabel('Count')
        for i, v in enumerate([survived_counts.get(0, 0), survived_counts.get(1, 0)]):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        axes[1].pie([survived_counts.get(0, 0), survived_counts.get(1, 0)], 
                   labels=['Not Survived (0)', 'Survived (1)'],
                   autopct='%1.1f%%',
                   colors=['#ff6b6b', '#4ecdc4'],
                   explode=(0.05, 0.05))
        axes[1].set_title('Survived Distribution (Pie Chart)')
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "survived_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"已成功绘制Survived列的分布图！\n图表已保存至: {save_path}\n\n统计信息:\n- 未生存(0): {survived_counts.get(0, 0)}人 ({survived_counts.get(0, 0)/len(df)*100:.1f}%)\n- 生存(1): {survived_counts.get(1, 0)}人 ({survived_counts.get(1, 0)/len(df)*100:.1f}%)"
    
    def _plot_age(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].hist(df['Age'].dropna(), bins=30, color='#45b7d1', edgecolor='white', alpha=0.7)
        axes[0].set_title('Age Distribution (Histogram)')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        
        df.boxplot(column='Age', by='Survived', ax=axes[1])
        axes[1].set_title('Age Distribution by Survived')
        axes[1].set_xlabel('Survived')
        axes[1].set_ylabel('Age')
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "age_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"已成功绘制Age列的分布图！\n图表已保存至: {save_path}"
    
    def _plot_pclass(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        pclass_counts = df['Pclass'].value_counts().sort_index()
        axes[0].bar(pclass_counts.index, pclass_counts.values, color=['#ff9f43', '#54a0ff', '#5f27cd'])
        axes[0].set_title('Pclass Distribution')
        axes[0].set_xlabel('Pclass')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks([1, 2, 3])
        
        survived_by_pclass = df.groupby('Pclass')['Survived'].mean()
        axes[1].bar(survived_by_pclass.index, survived_by_pclass.values, color=['#ff9f43', '#54a0ff', '#5f27cd'])
        axes[1].set_title('Survival Rate by Pclass')
        axes[1].set_xlabel('Pclass')
        axes[1].set_ylabel('Survival Rate')
        axes[1].set_xticks([1, 2, 3])
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "pclass_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"已成功绘制Pclass相关图表！\n图表已保存至: {save_path}"
    
    def _plot_sex(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sex_counts = df['Sex'].value_counts()
        axes[0].bar(sex_counts.index, sex_counts.values, color=['#ff6b6b', '#4ecdc4'])
        axes[0].set_title('Sex Distribution')
        axes[0].set_ylabel('Count')
        
        survived_by_sex = df.groupby('Sex')['Survived'].mean()
        axes[1].bar(survived_by_sex.index, survived_by_sex.values, color=['#ff6b6b', '#4ecdc4'])
        axes[1].set_title('Survival Rate by Sex')
        axes[1].set_ylabel('Survival Rate')
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "sex_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"已成功绘制Sex相关图表！\n图表已保存至: {save_path}"
    
    def _plot_overall(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        survived_counts = df['Survived'].value_counts()
        axes[0, 0].bar(['Not Survived', 'Survived'], 
                     [survived_counts.get(0, 0), survived_counts.get(1, 0)],
                     color=['#ff6b6b', '#4ecdc4'])
        axes[0, 0].set_title('Survived Distribution')
        axes[0, 0].set_ylabel('Count')
        
        pclass_counts = df['Pclass'].value_counts().sort_index()
        axes[0, 1].bar(pclass_counts.index, pclass_counts.values, color=['#ff9f43', '#54a0ff', '#5f27cd'])
        axes[0, 1].set_title('Pclass Distribution')
        axes[0, 1].set_xlabel('Pclass')
        axes[0, 1].set_ylabel('Count')
        
        axes[1, 0].hist(df['Age'].dropna(), bins=30, color='#45b7d1', edgecolor='white', alpha=0.7)
        axes[1, 0].set_title('Age Distribution')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(df['Fare'].dropna(), bins=30, color='#f9ca24', edgecolor='white', alpha=0.7)
        axes[1, 1].set_title('Fare Distribution')
        axes[1, 1].set_xlabel('Fare')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT_DIR, "overall_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f"已成功绘制数据总体分布图！\n图表已保存至: {save_path}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


class ModelTrainingInput(BaseModel):
    query: str = Field(description="用户关于模型训练和预测的问题")


class ModelTrainingTool(BaseTool):
    name: str = "model_training"
    description: str = "使用sklearn训练机器学习模型来预测Survived列。可以进行模型训练、评估和预测。"
    args_schema: Type[BaseModel] = ModelTrainingInput
    
    def _run(self, query: str) -> str:
        df = get_dataframe()
        ensure_output_dir()
        
        try:
            feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
            df_model = df[feature_cols + ['Survived']].copy()
            
            df_model['Age'] = df_model['Age'].fillna(df_model['Age'].median())
            df_model['Fare'] = df_model['Fare'].fillna(df_model['Fare'].median())
            
            X = df_model[feature_cols]
            y = df_model['Survived']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            feature_importance.plot(x='feature', y='importance', kind='bar', ax=axes[0], color='#45b7d1')
            axes[0].set_title('Feature Importance')
            axes[0].set_xlabel('Feature')
            axes[0].set_ylabel('Importance')
            axes[0].legend().remove()
            
            cm = confusion_matrix(y_test, y_pred)
            im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1].set_title('Confusion Matrix')
            axes[1].set_xticks([0, 1])
            axes[1].set_yticks([0, 1])
            axes[1].set_xticklabels(['Not Survived', 'Survived'])
            axes[1].set_yticklabels(['Not Survived', 'Survived'])
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Actual')
            
            for i in range(2):
                for j in range(2):
                    axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', 
                               color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
            
            plt.tight_layout()
            save_path = os.path.join(config.OUTPUT_DIR, "model_results.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result = "=== 模型训练结果 ===\n\n"
            result += f"使用模型: Random Forest Classifier\n"
            result += f"特征列: {', '.join(feature_cols)}\n"
            result += f"训练集大小: {len(X_train)}\n"
            result += f"测试集大小: {len(X_test)}\n\n"
            result += f"=== 模型性能 ===\n"
            result += f"准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)\n\n"
            result += f"=== 分类报告 ===\n"
            result += f"类别 0 (Not Survived):\n"
            result += f"  - Precision: {report['0']['precision']:.4f}\n"
            result += f"  - Recall: {report['0']['recall']:.4f}\n"
            result += f"  - F1-score: {report['0']['f1-score']:.4f}\n"
            result += f"类别 1 (Survived):\n"
            result += f"  - Precision: {report['1']['precision']:.4f}\n"
            result += f"  - Recall: {report['1']['recall']:.4f}\n"
            result += f"  - F1-score: {report['1']['f1-score']:.4f}\n\n"
            result += f"=== 特征重要性 ===\n"
            for _, row in feature_importance.iterrows():
                result += f"  {row['feature']}: {row['importance']:.4f}\n"
            result += f"\n图表已保存至: {save_path}"
            
            return result
            
        except Exception as e:
            return f"模型训练时发生错误: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


def get_tools():
    return [
        DataSummaryTool(),
        DataVisualizationTool(),
        ModelTrainingTool()
    ]
