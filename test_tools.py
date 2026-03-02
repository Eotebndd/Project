"""
测试脚本 - 直接测试三个核心工具功能
无需API Key即可验证工具是否正常工作
"""
import os
import sys

print("="*70)
print("泰坦尼克号数据探索Agent - 工具功能测试")
print("="*70)

from tools import DataSummaryTool, DataVisualizationTool, ModelTrainingTool

summary_tool = DataSummaryTool()
viz_tool = DataVisualizationTool()
model_tool = ModelTrainingTool()

print("\n" + "="*70)
print("【功能1: 数据摘要统计】")
print("="*70)
result1 = summary_tool._run("请给我数据的统计摘要")
print(result1)

print("\n" + "="*70)
print("【功能2: 数据可视化 - Survived列分布】")
print("="*70)
result2 = viz_tool._run("画出Survived列的分布")
print(result2)

print("\n" + "="*70)
print("【功能3: 模型训练与预测】")
print("="*70)
result3 = model_tool._run("训练一个模型预测Survived")
print(result3)

print("\n" + "="*70)
print("所有功能测试完成！")
print("生成的图表保存在 output_plots 目录中")
print("="*70)
