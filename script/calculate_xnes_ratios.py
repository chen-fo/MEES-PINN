#!/usr/bin/env python3
"""
脚本用于计算 xNES_Adam_no_jit 与 xNES_Adam 的时间比值
"""

import pandas as pd
from pathlib import Path

def main():
    # 读取CSV文件
    csv_path = Path("/home/chenfanke/TaskPINN/script/xnes_final_times_comparison.csv")
    
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}")
        return
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 计算比值
    df['Ratio (no_jit/jit)'] = df['xNES_Adam_no_jit'] / df['xNES_Adam']
    
    # 重新排列列顺序
    df = df[['Problem', 'xNES_Adam', 'xNES_Adam_no_jit', 'Ratio (no_jit/jit)']]
    
    # 保存到新的CSV文件
    output_path = Path("/home/chenfanke/TaskPINN/script/xnes_time_ratios.csv")
    df.to_csv(output_path, index=False)
    
    # 打印结果
    print("xNES-Adam 时间比值分析:")
    print("=" * 60)
    print(f"{'问题名称':<25} {'比值 (no_jit/jit)':<15}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        problem = row['Problem']
        ratio = row['Ratio (no_jit/jit)']
        print(f"{problem:<25} {ratio:<15.4f}")
    
    print("-" * 60)
    
    # 计算统计信息
    avg_ratio = df['Ratio (no_jit/jit)'].mean()
    min_ratio = df['Ratio (no_jit/jit)'].min()
    max_ratio = df['Ratio (no_jit/jit)'].max()
    
    print(f"统计信息:")
    print(f"  平均比值: {avg_ratio:.4f}")
    print(f"  最小比值: {min_ratio:.4f}")
    print(f"  最大比值: {max_ratio:.4f}")
    print(f"  总共问题数: {len(df)}")
    
    print(f"\n详细结果已保存到: {output_path}")

if __name__ == "__main__":
    main()