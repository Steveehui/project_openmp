# 实验数据说明

本目录用于存储实验运行产生的数据和图表。

## 文件说明

- `data.csv`: 原始实验数据(由 run_experiments.ps1 生成)
- `summary_table.txt`: 汇总统计表(由 plot_results.py 生成)
- `experiment1_scalability.png`: 实验1结果图表(强扩展性)
- `experiment2_problem_size.png`: 实验2结果图表(弱扩展性)
- `experiment3_comparison.png`: 实验3结果图表(策略对比)

## 使用流程

1. 运行实验: `.\scripts\run_experiments.ps1`
2. 生成图表: `python .\scripts\plot_results.py`
3. 查看结果: 打开 PNG 图表文件
