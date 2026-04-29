import pandas as pd
import os

# 读取CSV文件
def process_csv_file(input_file, output_file=None, output_folder=None):
    # 检查输入文件是否为CSV格式
    if not input_file.lower().endswith('.csv'):
        print("警告：输入文件不是CSV格式")
        return
    
    # 如果未指定输出文件，则在输入文件名后添加"_processed"后缀
    if output_file is None:
        file_name, file_ext = os.path.splitext(os.path.basename(input_file))
        output_file_name = f"{file_name}_processed{file_ext}"
    else:
        # 如果指定了输出文件名，只取文件名部分
        output_file_name = os.path.basename(output_file)
    
    # 如果指定了输出文件夹
    if output_folder is not None:
        # 确保输出文件夹存在，如果不存在则创建
        os.makedirs(output_folder, exist_ok=True)
        # 组合完整的输出文件路径
        output_file = os.path.join(output_folder, output_file_name)
    else:
        # 如果未指定输出文件夹，使用输入文件所在目录
        if output_file is None:
            output_file = os.path.join(os.path.dirname(input_file), output_file_name)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查是否存在route列
        if 'route' not in df.columns:
            print("警告：CSV文件中未找到'route'列")
            return
        
        # 要删除的route值列表
        routes_to_delete = ['A1', 'A2', 'A3', 'H']
        
        # 过滤出不需要删除的行
        filtered_df = df[~df['route'].isin(routes_to_delete)]
        
        # 计算删除的行数
        rows_deleted = len(df) - len(filtered_df)
        print(f"已删除 {rows_deleted} 行数据")
        
        # 保存处理后的CSV文件
        filtered_df.to_csv(output_file, index=False)
        print(f"处理完成，结果已保存至：{output_file}")
        
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

if __name__ == "__main__":
    # 用户需要修改为实际的CSV文件路径
    input_file = r"C:\Users\86180\Desktop\大三\低空经济\无人机数据\flights.csv"  # 替换为你的CSV文件路径
    # 指定输出文件夹路径（使用原始字符串r前缀，避免反斜杠转义问题）
    output_folder = r"C:\Users\86180\Desktop\研究\无人机能耗\data"
    process_csv_file(input_file, output_folder=output_folder)