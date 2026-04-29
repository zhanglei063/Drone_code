import os
import pandas as pd

# 处理CSV文件，计算时间间隔、功率和能耗
def process_flight_data(input_file):
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        required_columns = ['flight', 'time', 'battery_voltage', 'battery_current']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告：CSV文件中缺少以下必要列: {', '.join(missing_columns)}")
            return
        
        # 确保time列是数字类型
        if not pd.api.types.is_numeric_dtype(df['time']):
            try:
                df['time'] = pd.to_numeric(df['time'])
                print("已将time列转换为数字类型")
            except ValueError:
                print("错误：无法将time列转换为数字类型")
                return
        
        # 按flight列分组
        grouped = df.groupby('flight')
        
        # 创建一个结果DataFrame来存储处理后的数据
        result_df = pd.DataFrame()
        
        # 对每个flight分组进行处理
        for flight_id, group in grouped:
            # 复制当前组的数据
            processed_group = group.copy()
            
            # 按time列排序
            processed_group = processed_group.sort_values('time')
            
            # 计算时间间隔（对于数字类型的time列）
            processed_group['time_diff'] = processed_group['time'].diff().round(6)
            
            # 第一行的时间差设为0
            if not processed_group.empty:
                processed_group.loc[processed_group.index[0], 'time_diff'] = 0
            
            # 计算功率（battery_voltage * battery_current）
            processed_group['power'] = processed_group['battery_voltage'] * processed_group['battery_current']
            
            # 计算能耗（功率 * 时间间隔）
            processed_group['energy_consumption'] = processed_group['power'] * processed_group['time_diff']
            # 处理可能的NaN值（如果time_diff或power为NaN）
            processed_group['energy_consumption'] = processed_group['energy_consumption'].fillna(0)
            
            # 将处理后的组添加到结果DataFrame中
            result_df = pd.concat([result_df, processed_group])
        
        # 确保结果保存在指定的data文件夹中
        output_folder = r"C:\Users\86180\Desktop\研究\无人机能耗\data"
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取输入文件名，并创建输出文件名
        input_file_name = os.path.basename(input_file)
        file_name, file_ext = os.path.splitext(input_file_name)
        output_file = os.path.join(output_folder, f"{file_name}_energy_consumption{file_ext}")
        
        # 保存处理后的文件
        result_df.to_csv(output_file, index=False)
        print(f"处理完成，结果已保存至：{output_file}")
        
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

if __name__ == "__main__":
    # 用户需要修改为实际的CSV文件路径（处理后的文件）
    input_file = r"C:\Users\86180\Desktop\研究\无人机能耗\data\flights_processed.csv"  # 替换为你的处理后的CSV文件路径
    process_flight_data(input_file)