# -*- coding: utf-8 -*-
import json

import pandas as pd

jod_dict = {"0": "other",
            "1": "academic/educator",
            "2": "artist",
            "3": "clerical/admin",
            "4": "college/grad student",
            "5": "customer service",
            "6": "doctor/health care",
            "7": "executive/managerial",
            "8": "farmer",
            "9": "homemaker",
            "10": "K-12 student",
            "11": "lawyer",
            "12": "programmer",
            "13": "retired",
            "14": "sales/marketing",
            "15": "scientist",
            "16": "self-employed",
            "17": "technician/engineer",
            "18": "tradesman/craftsman",
            "19": "unemployed",
            "20": "writer"}

age_dict = {"1": "under 18",
            "18": "18-24",
            "25": "25-34",
            "35": "35-44",
            "45": "45-49",
            "50": "50-55",
            "56": "56+"}

gender_dict = {"F": "Female",
               "M": "Male"}


def get_zipcode_dict(json_file="proc_data/zipcode_dict.json"):
    zipcode_dict = json.load(open(json_file, 'r'))
    return zipcode_dict


def data_preprocess_for_llm_embedding(raw_data, max_hist=5, save_path="proc_data/llm_data_for_embedding.csv"):
    zipcode_dict = get_zipcode_dict()
    hist_dict = {}
    hist_interaction = []
    for i, item in raw_data.iterrows():
        u_id = item['User ID']
        title = item['Title']
        rating = item['Rating']

        if u_id not in hist_dict:
            hist_dict[u_id] = []
        hist = hist_dict[u_id][-max_hist:]
        if len(hist) < 1:
            hist_str = "Unknown"
        else:
            hist_str = ";".join(["%s,%s" % (m[0], m[1]) for idx, m in enumerate(hist)])

        hist_interaction.append(hist_str)
        hist_dict[u_id].append((title, rating))

    raw_data['Hist'] = hist_interaction
    raw_data['Gender'] = raw_data['Gender'].map(gender_dict)
    raw_data['Age'] = raw_data['Age'].map(age_dict)
    raw_data['Job'] = raw_data['Job'].map(jod_dict)
    raw_data['Address'] = raw_data['Zipcode'].map(zipcode_dict)

    raw_data.to_csv(save_path, index=False)



if __name__ == '__main__':
    raw_file = "proc_data_all/sampled_data.csv"
    raw_data = pd.read_csv(raw_file, dtype=str)
    data_preprocess_for_llm_embedding(raw_data, save_path="proc_data/llm_data_for_embedding.csv")

