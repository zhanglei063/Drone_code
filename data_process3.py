import pandas as pd
import os

def process_energy_consumption(input_file, output_file=None):
    """
    处理数据文件，将energy_consumption列取绝对值，并删除battery_current为负数的行
    支持Excel和CSV格式，根据文件扩展名自动选择
    
    参数:
    input_file: 输入文件路径
    output_file: 输出文件路径，如果为None则覆盖原文件
    """
    # 如果未指定输出文件，则覆盖原文件
    if output_file is None:
        output_file = input_file
    
    try:
        # 获取文件扩展名，转换为小写
        file_extension = os.path.splitext(input_file)[1].lower()
        
        # 根据文件扩展名选择合适的读取方法
        if file_extension == '.csv':
            # 添加low_memory=False参数来避免混合类型警告
            df = pd.read_csv(input_file, low_memory=False)
        elif file_extension in ['.xlsx', '.xls']:
            # 对于Excel文件，指定引擎
            df = pd.read_excel(input_file, engine='openpyxl')
        else:
            print(f"不支持的文件格式: {file_extension}")
            return
        
        # 检查是否存在energy_consumption列
        if 'energy_consumption' not in df.columns:
            print(f"警告: 文件 {input_file} 中不存在 'energy_consumption' 列")
            return
        
        # 将energy_consumption列的值取绝对值
        df['energy_consumption'] = df['energy_consumption'].abs()
        
        # 检查并处理battery_current列
        if 'battery_current' in df.columns:
            # 记录删除前的行数
            before_rows = len(df)
            # 删除battery_current为负数的行
            df = df[df['battery_current'] >= 0]
            # 计算删除的行数
            deleted_rows = before_rows - len(df)
            print(f"已删除 {deleted_rows} 行battery_current为负数的数据")
        else:
            print(f"警告: 文件中不存在 'battery_current' 列，跳过相关处理")
        
        # 根据输出文件扩展名选择合适的保存方法
        output_extension = os.path.splitext(output_file)[1].lower()
        if output_extension == '.csv':
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        elif output_extension in ['.xlsx', '.xls']:
            # 对于Excel文件，指定引擎
            df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            print(f"不支持的输出文件格式: {output_extension}")
            return
        
        print(f"处理完成! 结果已保存至: {output_file}")
        print(f"处理后的数据行数: {len(df)}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

if __name__ == "__main__":
    # 示例用法 - CSV文件
    sample_input_file = r"C:\Users\86180\Desktop\研究\无人机能耗\data\flights_processed_energy_consumption.csv"
    sample_output_file = r"C:\Users\86180\Desktop\研究\无人机能耗\data\flights_processed_energy_consumption3.csv"
    
    # 使用方法2: 保存为新文件
    process_energy_consumption(sample_input_file, sample_output_file)
    
    print("数据处理完毕")