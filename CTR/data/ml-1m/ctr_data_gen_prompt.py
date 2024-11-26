import pandas as pd
import json
import pickle
import csv


PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<Task>
You are a seasoned movie recommender with expertise in predicting user preferences based on their past ratings. 
Given the following user information and his/her movie rating history, and a new movie with title and genre. 
Please decide wether the user will enjoy the new movie:

<User Information>
{user_info}

<User History>
{user_hist}

<New Movie Information>
{item_info}

Please ONLY output YES or NO to decide wether the user will enjoy the new movie:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT2 = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<Task>
You are a seasoned movie recommender with expertise in predicting user preferences based on their past ratings. Given the following user information  and his/her movie rating history, suggest ONE movie he/she would like enjoy:
<User Information>
{user_info}

<User Movie Ratings>
{user_hist}
Please only output the movie title without any other unrelated information.
Now, your movie recommendation is:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


PROMPT3 = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<Task>
You are a seasoned movie recommender with expertise in predicting user preferences based on their past ratings. Given the following user information  and his/her movie rating history, suggest ONE movie he/she would like enjoy:
<User Information>
{user_info}

<User Movie Ratings>
{user_hist}
Please only output the movie title without any other unrelated information.
Now, your movie recommendation is:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def table_to_markdown(data, headers=None):
    if not data or data=='Unknown' or len(data)==0:
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
    movie_list = []
    if movie_string is None or movie_string == "Unknown":
        return movie_list

    movies = [movie.split("|") for movie in movie_string.split("#")]
    for movie in movies:
        movie[0] = movie[0].split("(")[0].strip()
        movie_list.append(movie)
    return movie_list

def movie_hist_list(movie_hist):
    if movie_hist is None or movie_hist == "Unknown":
        return []
    hist_info = [movie.split(",") for movie in movie_hist.split(";") if len(movie.split(","))==2]
    return hist_info


def prompt_gen(row):
    user_header = ["Gender","Age","Job", "Address"]
    user_info = [row[5], row[6], row[7], row[14], row[13]]
    user_data = table_to_markdown([user_header, user_info])

    hist_header = ["Title","Rating"]
    hist_info = movie_hist_list(row[13])
    if len(hist_info) == 0:
        user_hist = 'No history'
    else:
        hist_data = [hist_header] + hist_info
        user_hist = table_to_markdown(hist_data)
    
    item_header = ["Title","Genres"]
    item_info = [row[9], row[10]]
    if len(item_info) > 50:
        item_info = item_info[-50:]
    item_data = table_to_markdown([item_header, item_info])

    prompt = PROMPT.format(user_info=user_data, user_hist=user_hist, item_info=item_data)
    # prompt = PROMPT2.format(user_info=user_data, user_hist=user_hist)

    return prompt


def ctr_data_gen_prompt(raw_data_path="proc_data/llm_data_for_embedding.csv", save_path="proc_data/ml_data_with_prompt.pkl"):

    with open(raw_data_path,'r',encoding='utf-8') as data_file:
        datas = pd.read_csv(data_file)
        datas['prompt'] = datas.apply(prompt_gen, axis=1)
            

    with open(save_path, 'wb') as resfile:
        # print(datas['emb'])
        pickle.dump(datas, resfile)


if __name__ == '__main__':
    ctr_data_gen_prompt()
