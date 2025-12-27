#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像卷积实验结果可视化脚本
生成加速比、并行效率等图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体(Windows 环境)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_FILE = RESULTS_DIR / "data.csv"

def load_data():
    """加载实验数据"""
    if not DATA_FILE.exists():
        print(f"错误: 未找到数据文件 {DATA_FILE}")
        print("请先运行 run_experiments.ps1 生成数据")
        return None
    
    df = pd.read_csv(DATA_FILE, encoding='utf-8')
    print(f"成功加载 {len(df)} 条实验记录")
    return df

def plot_experiment1(df):
    """实验1: 强扩展性(变线程数)"""
    exp1 = df[df['实验组'] == '实验1']
    
    if exp1.empty:
        print("警告: 实验1 数据为空")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 加速比曲线
    for version in exp1['版本'].unique():
        data = exp1[exp1['版本'] == version]
        ax1.plot(data['线程数'], data['加速比'], marker='o', label=version, linewidth=2)
    
    # 理想加速比(线性)
    threads = exp1['线程数'].unique()
    ax1.plot(threads, threads, 'k--', label='理想线性加速', alpha=0.5)
    
    ax1.set_xlabel('线程数', fontsize=12)
    ax1.set_ylabel('加速比', fontsize=12)
    ax1.set_title('实验1: 强扩展性测试 (固定负载)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 并行效率
    for version in exp1['版本'].unique():
        if version == 'Serial':
            continue
        data = exp1[exp1['版本'] == version]
        efficiency = (data['加速比'] / data['线程数']) * 100
        ax2.plot(data['线程数'], efficiency, marker='s', label=version, linewidth=2)
    
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='理想效率')
    ax2.set_xlabel('线程数', fontsize=12)
    ax2.set_ylabel('并行效率 (%)', fontsize=12)
    ax2.set_title('并行效率 = 加速比 / 线程数 × 100%', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'experiment1_scalability.png', dpi=300, bbox_inches='tight')
    print(f"已保存: experiment1_scalability.png")
    plt.close()

def plot_experiment2(df):
    """实验2: 弱扩展性(变问题规模)"""
    exp2 = df[df['实验组'] == '实验2']
    
    if exp2.empty:
        print("警告: 实验2 数据为空")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 运行时间对比
    sizes = exp2['图像尺寸'].unique()
    x = np.arange(len(sizes))
    width = 0.35
    
    for i, version in enumerate(exp2['版本'].unique()):
        data = exp2[exp2['版本'] == version]
        times = [data[data['图像尺寸'] == s]['运行时间(秒)'].values[0] for s in sizes]
        ax1.bar(x + i * width, times, width, label=version)
    
    ax1.set_xlabel('图像尺寸', fontsize=12)
    ax1.set_ylabel('运行时间 (秒)', fontsize=12)
    ax1.set_title('实验2: 不同图像尺寸的运行时间对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 加速比对比
    for version in exp2['版本'].unique():
        if version == 'Serial':
            continue
        data = exp2[exp2['版本'] == version]
        ax2.plot(data['图像尺寸'], data['加速比'], marker='o', label=version, linewidth=2)
    
    ax2.set_xlabel('图像尺寸', fontsize=12)
    ax2.set_ylabel('加速比', fontsize=12)
    ax2.set_title('加速比随问题规模的变化', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'experiment2_problem_size.png', dpi=300, bbox_inches='tight')
    print(f"已保存: experiment2_problem_size.png")
    plt.close()

def plot_experiment3(df):
    """实验3: 四种策略对比"""
    exp3 = df[df['实验组'] == '实验3']
    
    if exp3.empty:
        print("警告: 实验3 数据为空")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    versions = exp3['版本'].tolist()
    times = exp3['运行时间(秒)'].tolist()
    speedups = exp3['加速比'].tolist()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # 运行时间柱状图
    bars1 = ax1.bar(versions, times, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('运行时间 (秒)', fontsize=12)
    ax1.set_title('实验3: 四种策略运行时间对比', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上标注数值
    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=10)
    
    # 加速比柱状图
    bars2 = ax2.bar(versions, speedups, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('加速比', fontsize=12)
    ax2.set_title('相对串行版本的加速比', fontsize=14, fontweight='bold')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='串行基准')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # 在柱子上标注数值
    for bar, speedup in zip(bars2, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'experiment3_comparison.png', dpi=300, bbox_inches='tight')
    print(f"已保存: experiment3_comparison.png")
    plt.close()

def generate_summary_table(df):
    """生成汇总表格"""
    summary = df.groupby(['实验组', '版本']).agg({
        '运行时间(秒)': ['mean', 'std'],
        '加速比': 'max'
    }).round(3)
    
    summary_file = RESULTS_DIR / 'summary_table.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验结果汇总表\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n" + "=" * 70 + "\n")
    
    print(f"已保存: summary_table.txt")
    print("\n汇总统计:")
    print(summary)

def main():
    print("=" * 50)
    print("  图像卷积实验结果可视化")
    print("=" * 50)
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 创建图表
    print("\n生成图表...")
    plot_experiment1(df)
    plot_experiment2(df)
    plot_experiment3(df)
    
    # 生成汇总表
    generate_summary_table(df)
    
    print("\n" + "=" * 50)
    print("  所有图表已生成!")
    print(f"  保存位置: {RESULTS_DIR}")
    print("=" * 50)

if __name__ == '__main__':
    main()
