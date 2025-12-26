#!/usr/bin/env python3
"""
脚本用于提取 multi_train_log/log_6 中 CMAES 和 CMAES_no_jit 方法的最终训练时间，
并生成对比CSV表格。
"""

import os
import csv
import pandas as pd
from pathlib import Path

# 定义路径
BASE_PATH = Path("/home/chenfanke/TaskPINN/multi_train_log/log_6")
METHODS = ["CMAES", "CMAES_no_jit"]
CSV_DIR = "loss_time_csv"

def extract_final_time(csv_file_path):
    """从CSV文件中提取最后一行的cum_time值"""
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        # 获取最后一行的cum_time值
        final_time = df.iloc[-1]['cum_time']
        return final_time
    except Exception as e:
        print(f"处理文件 {csv_file_path} 时出错: {e}")
        return None

def main():
    # 创建结果存储字典
    results = {}
    
    # 遍历每种方法
    for method in METHODS:
        method_path = BASE_PATH / method / CSV_DIR
        if not method_path.exists():
            print(f"路径不存在: {method_path}")
            continue
            
        results[method] = {}
        
        # 遍历该方法下的所有CSV文件
        for csv_file in method_path.glob("*.csv"):
            # 获取文件名（不含扩展名）
            problem_name = csv_file.stem.replace("_IterTime_Loss", "").replace("8*8", "")
            # 提取最终时间
            final_time = extract_final_time(csv_file)
            if final_time is not None:
                results[method][problem_name] = final_time
            else:
                results[method][problem_name] = "ERROR"
    
    # 创建DataFrame用于输出
    # 获取所有问题名称
    all_problems = set()
    for method in results:
        all_problems.update(results[method].keys())
    all_problems = sorted(list(all_problems))
    
    # 构建数据表格
    data = []
    for problem in all_problems:
        row = {"Problem": problem}
        for method in METHODS:
            row[method] = results[method].get(problem, "N/A")
        data.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到CSV文件
    output_path = Path("/home/chenfanke/TaskPINN/script/final_times_comparison.csv")
    df.to_csv(output_path, index=False)
    
    # 打印结果
    print("最终训练时间对比:")
    print(df.to_string(index=False))
    print(f"\n结果已保存到: {output_path}")
    
    # 计算平均时间比率
    if "CMAES" in results and "CMAES_no_jit" in results:
        ratios = []
        for problem in all_problems:
            if (problem in results["CMAES"] and 
                problem in results["CMAES_no_jit"] and 
                isinstance(results["CMAES"][problem], (int, float)) and 
                isinstance(results["CMAES_no_jit"][problem], (int, float))):
                ratio = results["CMAES_no_jit"][problem] / results["CMAES"][problem]
                ratios.append(ratio)
                print(f"{problem}: CMAES_no_jit/CMAES = {ratio:.2f}x")
        
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"\n平均时间比率 (CMAES_no_jit/CMAES): {avg_ratio:.2f}x")

if __name__ == "__main__":
    main()