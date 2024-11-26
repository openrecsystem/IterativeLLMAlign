import os
import re
import csv
import json
import random

PROMPT = """You are a seasoned movie recommender with expertise in predicting user preferences based on their past ratings. Given the following user information  and his/her movie rating history, suggest ONE movie he/she would like enjoy:
<User Information>
{user_info}

<User Movie Ratings>
{item_info}
Please only output the movie title without any other unrelated information.
Now, your movie recommendation is:
"""


def table_to_markdown(data, headers=None):
    if not data:
        return ""
    
    if headers is None:
        headers = data[0]
        data = data[1:]
    
    column_widths = [
        max(len(str(row[i])) for row in data + [headers])
        for i in range(len(headers))
    ]
    
    #Format table header
    markdown_table = "|" + "|".join(
        f" {header:<{width}} " for header, width in zip(headers, column_widths)
    ) + "|\n"
    markdown_table += "|" + "|".join("_" * (width + 2) for width in column_widths) + "\n"
    
    #Format table rows
    for row in data:
        markdown_table += (
            "|"
            + "|".join(f" {str(cell):<{width}} " for cell, width in zip(row, column_widths))
            + "|\n"
        )
    
    return markdown_table

def parse_movie_string(movie_string):
    movies = [movie.split("|") for movie in movie_string.split("#")]
    movie_list = []
    for movie in movies:
        movie[0] = movie[0].split("(")[0].strip()
        movie_list.append(movie)
    return movie_list
    
def split_array(arr):
    greater_than_3_idx = -1
    less_than_3_idx = -1
    accept = ""
    reject = ""
    #print(arr)
    for i in range(len(arr) -1, -1, -1):
        if len(arr[i]) != 3:
            del arr[i]
        if int(arr[i][2]) > 3 and greater_than_3_idx == -1:
            greater_than_3_idx = i
            accept = arr[i][0]
        if int(arr[i][2]) < 3 and less_than_3_idx == -1:
            less_than_3_idx = i
            reject = arr[i][0]
        if greater_than_3_idx != -1 and less_than_3_idx != -1:
            break
    if greater_than_3_idx != -1 and less_than_3_idx != -1:
        return arr[0:min(greater_than_3_idx, less_than_3_idx)], accept, reject
    else:
        return arr, "",""
   
def train_process(inputPath, outputPath):
    result_list = []
    with open(inputPath,'r',encoding='utf-8') as r_file:
        csvreader = csv.reader(r_file)
        next(csvreader)
        for row in csvreader:
            user_header = ["Gender","Age","Job"]
            user_info = [row[1], row[2], row[3]]
            user_data = table_to_markdown([user_header, user_info])
            #print("user_data:\n",user_data)
            
            item_header = ["Title","Genres","Rating"]
            item_info = parse_movie_string(row[6])
            item_info, accept, reject = split_array(item_info)
            if "" == accept and len(item_info) < 2:
                continue
            item_data =[item_header]
            filter_item_info = [item for item in item_info if len(item) == 3 and int(item[2]) >= 3]
            if len(filter_item_info) > 50:
                filter_item_info = filter_item_info[-50:]
                
            for item in filter_item_info:
                item_data.append(item)
            try:
                item_data = table_to_markdown(item_data)
            except:
                continue
            #print("item_data:\n", item_data)
            prompt = PROMPT.format(user_info = user_data, item_info = item_data)
            
            conversion_value = []
            conversion_value.append({"from":"human","value":prompt})
            chosen_value = {"from":"gpt","value":accept}
            reject_value = {"from":"gpt","value":reject}
            result = {"conversations":conversion_value,"chosen":chosen_value,"rejected":reject_value}
            result_list.append(result)
        print("sample size:", len(result_list))
        with open(outputPath,'w',encoding='utf-8') as w_file:
            json.dump(result_list, w_file, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':
    inputPath = "/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/llm_data_train.csv"
    outputPath = "/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/dpo_movielens.json"
    train_process(inputPath, outputPath)