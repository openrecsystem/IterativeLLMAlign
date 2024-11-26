# -*- coding: utf-8 -*-
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

job_dict = {"0": "other",
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


def data_preprocess_for_llm_train(data, save_path="proc_data/llm_data.csv"):
    zipcode_dict = get_zipcode_dict()

    group_data = data.groupby('User ID')

    user_ids = []
    user_genders = []
    user_ages = []
    user_jobs = []
    user_zipcodes = []
    user_address = []
    user_type = []

    user_hist = []

    for user_id, group in group_data:
        user_ids.append(user_id)

        hist_info = []
        for i, item in group.iterrows():
            u_gender = item['Gender']
            u_age = item['Age']
            u_job = item['Job']
            u_zipcode = item['Zipcode']
            u_type = item['type']

            movie_name = item['Title']
            genre = item['Genres'].replace("|", ",")
            rating = item['Rating']
            info = f"{movie_name}|{genre}|{rating}"
            hist_info.append(info)

        user_genders.append(gender_dict[u_gender])
        user_ages.append(age_dict[u_age])
        user_jobs.append(job_dict[u_job])
        user_zipcodes.append(u_zipcode)
        user_address.append(zipcode_dict[u_zipcode])
        user_hist.append('#'.join(hist_info))
        user_type.append(u_type)
    
    df = pd.DataFrame(
        {'User ID': user_ids, 'Gender': user_genders, 'Age': user_ages, 'Job': user_jobs, 'Zipcode': user_zipcodes,
         'Address': user_address, 'Hist': user_hist, 'type': user_type})
    df.to_csv(save_path, index=False)

    return df


def split_train_and_test_data(df, train_data_path="proc_data/llm_data_train.csv", test_data_path="proc_data/llm_data_test.csv"):
    
    print(len(df))

    train_data = df[df['type'] == '1']
    test_data = df[df['type'] == '2']

    print(len(train_data))
    print(len(test_data))

    test_hist_movie = []
    test_next_movie = []

    for i, item in test_data.iterrows():
        hist = item['Hist'].split('#')
        test_next_movie.append(hist[-1])
        hist = hist[:-1]
        test_hist_movie.append("#".join(hist))

    test_data['Hist'] = test_hist_movie
    test_data['Next'] = test_next_movie


    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)



if __name__ == '__main__':
    data = pd.read_csv('proc_data_all/sampled_data.csv', dtype=str) 
    df = data_preprocess_for_llm_train(data, save_path="proc_data/llm_data.csv")
    split_train_and_test_data(df, train_data_path="proc_data/llm_data_train.csv", test_data_path="proc_data/llm_data_test.csv")
    

    
    
