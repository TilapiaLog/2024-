import pandas as pd
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fill_missing_with_mode(df):
    for column in df.columns:
        mode_value = df[column].mode()
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value[0])  # 使用赋值方式填充缺失值
    return df

def process_data(train_data_path,  target_data_path, output_prefix='filled'):
    try:
        # 读取数据
        train_data = pd.read_csv(train_data_path)
        target_data = pd.read_csv(target_data_path)
        logging.info("数据读取成功。")
    except FileNotFoundError as e:
        logging.error(f"文件未找到: {e}")
        return
    except pd.errors.EmptyDataError:
        logging.error("读取的文件为空。")
        return
    except Exception as e:
        logging.error(f"读取数据时发生错误: {e}")
        return

    # 删除重复行
    train_data.drop_duplicates(inplace=True)

    logging.info("已删除重复行。")

    # 填充缺失值
    train_data = fill_missing_with_mode(train_data)

    logging.info("已填充缺失值。")

    # 保存处理后的数据
    train_data.to_csv(f'{output_prefix}_train_data.csv', index=False)
    target_data.to_csv(f'{output_prefix}_train_target.csv', index=False)
    logging.info("处理后的数据已保存。")

if __name__ == "__main__":
    # 定义文件路径
    train_data_path = os.path.join('train', 'train_data.csv')
    target_data_path = os.path.join('train', 'train_target.csv')

    process_data(train_data_path, target_data_path)