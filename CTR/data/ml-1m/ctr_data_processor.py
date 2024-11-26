# -*- coding: utf-8 -*-
import pandas as pd
import json

def get_meta_data_and_ctr_data(raw_data, save_json_path="proc_data/ml-1m-meta.json", save_path="proc_data/ctr_data.csv"):

    field_names = ["Gender", "Age", "Job", "Zipcode", "Movie ID", "Title", "First genre", "User ID"]

    # 计算每个字段的特征计数
    feature_count = [raw_data[field].nunique() for field in field_names]

    # 生成特征字典
    feature_dict = {field: {str(val): idx for idx, val in enumerate(raw_data[field].unique())} for field in field_names}

    # 计算特征偏移量
    feature_offset = [0]  # 初始化第一个特征偏移量为0
    for count in feature_count[:-1]:
        feature_offset.append(feature_offset[-1] + count)

    for field in field_names:
        raw_data[field] = raw_data[field].map(feature_dict[field])
        

    # 生成meta_data字典
    meta_data = {
        'field_names': field_names,
        'feature_count': feature_count,
        'feature_dict': feature_dict,
        'feature_offset': feature_offset
    }

    # 将meta_data字典保存为JSON文件
    with open(save_json_path, 'w') as json_file:
        json.dump(meta_data, json_file, indent=4)

    raw_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    raw_file = "proc_data/sampled_data.csv"
    raw_data = pd.read_csv(raw_file, dtype=str)
    get_meta_data_and_ctr_data(raw_data, save_json_path="proc_data/ml-1m-meta.json", save_path="proc_data/ctr_data.csv")





