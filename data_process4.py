import pandas as pd
import os

def calculate_unit_energy(input_file, output_file=None):
    """
    处理CSV文件，计算单位能耗（energy_consumption / time_diff）
    处理time_diff为0的情况，避免除以零错误
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径，如果为None则自动生成
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        required_columns = ['energy_consumption', 'time_diff']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告：CSV文件中缺少以下必要列: {', '.join(missing_columns)}")
            return
        
        # 计算单位能耗，处理time_diff为0的情况
        # 使用numpy的where函数：当time_diff不为0时计算，否则设为0
        df['unit_energy_consumption'] = pd.np.where(
            df['time_diff'] != 0,  # 条件
            df['energy_consumption'] / df['time_diff'],  # 条件成立时的值
            0  # 条件不成立时的值（time_diff为0）
        )
        
        # 如果未指定输出文件，则自动生成
        if output_file is None:
            file_name, file_ext = os.path.splitext(os.path.basename(input_file))
            output_file = os.path.join(os.path.dirname(input_file), f"{file_name}_with_unit{file_ext}")
        
        # 保存处理后的文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 统计信息
        total_rows = len(df)
        zero_time_diff_rows = (df['time_diff'] == 0).sum()
        print(f"处理完成！结果已保存至: {output_file}")
        print(f"总数据行数: {total_rows}")
        print(f"time_diff为0的行数: {zero_time_diff_rows} (这些行的单位能耗已设为0)")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    # 用户需要修改为实际的CSV文件路径
    input_file = r"C:\Users\86180\Desktop\研究\无人机能耗\data\flights_processed_energy_consumption3.csv"
    output_file = r"C:\Users\86180\Desktop\研究\无人机能耗\data\flights_processed_energy_consumption4.csv"
    
    calculate_unit_energy(input_file, output_file)
