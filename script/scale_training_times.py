#!/usr/bin/env python3
"""
脚本用于将训练时间结果乘以50，可能用于不同的迭代次数或其他缩放需求。
"""

import pandas as pd
from pathlib import Path

def main():
    # 读取原始结果文件
    input_path = Path("/home/chenfanke/TaskPINN/script/final_times_comparison.csv")
    
    if not input_path.exists():
        print(f"输入文件不存在: {input_path}")
        return
    
    # 读取CSV文件
    df = pd.read_csv(input_path)
    
    # 创建新的DataFrame，将数值列乘以50
    scaled_df = df.copy()
    
    # 将CMAES和CMAES_no_jit列乘以50
    scaled_df['CMAES'] = scaled_df['CMAES'] * 50
    scaled_df['CMAES_no_jit'] = scaled_df['CMAES_no_jit'] * 50
    
    # 重新计算比率列
    scaled_df['Ratio (no_jit/jit)'] = scaled_df['CMAES_no_jit'] / scaled_df['CMAES']
    
    # 保存缩放后的结果
    output_path = Path("/home/chenfanke/TaskPINN/script/scaled_final_times_comparison.csv")
    scaled_df.to_csv(output_path, index=False)
    
    # 打印缩放后的结果
    print("=" * 90)
    print("缩放后的训练时间对比 (所有时间乘以50)")
    print("=" * 90)
    print(f"{'问题名称':<25} {'CMAES (秒)':<15} {'CMAES_no_jit (秒)':<20} {'倍数':<10}")
    print("-" * 90)
    
    for _, row in scaled_df.iterrows():
        problem = row['Problem']
        cmaes_time = row['CMAES']
        no_jit_time = row['CMAES_no_jit']
        ratio = row['Ratio (no_jit/jit)']
        
        print(f"{problem:<25} {cmaes_time:<15.2f} {no_jit_time:<20.2f} {ratio:<10.2f}x")
    
    print("-" * 90)
    
    # 计算统计信息
    avg_ratio = scaled_df['Ratio (no_jit/jit)'].mean()
    min_ratio = scaled_df['Ratio (no_jit/jit)'].min()
    max_ratio = scaled_df['Ratio (no_jit/jit)'].max()
    
    print(f"统计信息:")
    print(f"  平均时间比率 (CMAES_no_jit/CMAES): {avg_ratio:.2f}x")
    print(f"  最小时间比率: {min_ratio:.2f}x")
    print(f"  最大时间比率: {max_ratio:.2f}x")
    print(f"  总共分析问题数: {len(scaled_df)}")
    
    print(f"\n缩放后的结果已保存到: {output_path}")

if __name__ == "__main__":
    main()