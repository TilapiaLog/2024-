import pandas as pd
import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def drop_rows_with_high_missing(df, threshold=0.3):
    # 计算每行缺失值的比例
    missing_ratio = df.isnull().mean(axis=1)
    # 删除缺失值比例超过阈值的行
    df_cleaned = df[missing_ratio < threshold]
    return df_cleaned

def process_data(train_data_path, target_data_path, output_prefix='deleted'):
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
    target_data.drop_duplicates(inplace=True)
    logging.info("已删除重复行。")

    # 删除缺失值比例超过50%的行
    train_data = drop_rows_with_high_missing(train_data)
    target_data = drop_rows_with_high_missing(target_data)
    logging.info("已删除缺失值比例超过50%的行。")

    # 保存处理后的数据
    train_data.to_csv(f'{output_prefix}_train_data.csv', index=False)
    target_data.to_csv(f'{output_prefix}_train_target.csv', index=False)
    logging.info("处理后的数据已保存。")

if __name__ == "__main__":
    # 定义文件路径
    train_data_path = os.path.join('train', 'train_data.csv')
    target_data_path = os.path.join('train', 'train_target.csv')

    process_data(train_data_path, target_data_path)