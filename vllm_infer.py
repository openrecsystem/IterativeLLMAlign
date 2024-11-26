import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import argparse
import csv
from vllm import LLM, SamplingParams

os.environ["OMP_NUM_THREADS"] = "16"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PROMPT = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<Task>
You are a seasoned movie recommender with expertise in predicting user preferences based on their past ratings. 
Given the following user information and his/her movie rating history, suggest FIVE movies he/she would like to enjoy:
<User Information>
{user_info}

<User Movie Ratings>
{item_info}
Please only output the movie title without any other unrelated information.
Now, your movie recommendations are:<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def extract_movie_names(text):
    # pattern = r"(?:\d+\.\s*)?([^\n]+?) \(\d{4}\)"
    pattern = r'\d+\.\s*(.*?)\n'
    movie_names = re.findall(pattern, text)

    return movie_names


def movie_words_match(gen_movie, next_movie):
    if next_movie is not None:
        gen_words = set(gen_movie.replace(',', ' ').split())
        next_words = set(next_movie.replace(',', ' ').split())
        
        discard_words = ['The', 'the', 'one', 'A', 'An', 'and', 'And', 'of', 'Of']
        for word in discard_words:
            gen_words.discard(word)
            next_words.discard(word)
        
        return next_words.issubset(gen_words) or gen_words.issubset(next_words)

    else:
        return False
            


def movie_match(row):
    match_num = 0
    
    true_movie = [ movie.split('|')[0].split('(')[0] if movie.split('|')[2]>'3' else None for movie in row['Next'].split("#") ]

    for movie in true_movie:
        if movie is not None:
            gen_words = set(row["gen_movie"].replace(',', ' ').split())
            next_words = set(movie.replace(',', ' ').split())
            
            discard_words = ['The', 'the', 'one', 'A', 'An', 'and', 'And', 'of', 'Of']
            for word in discard_words:
                gen_words.discard(word)
                next_words.discard(word)
            
            if next_words.issubset(gen_words):
                match_num += 1
        
    return match_num


def movie_dismatch(row):
    match_num = 0
    
    true_movie = [ movie.split('|')[0].split('(')[0] if movie.split('|')[2]<'3' else None for movie in row['Next'].split("#") ]
    
    for movie in true_movie:
        if movie is not None:
            gen_words = set(row["gen_movie"].replace(',', ' ').split())
            next_words = set(movie.replace(',', ' ').split())
            
            discard_words = ['The', 'the', 'one', 'A', 'An', 'and', 'And', 'of', 'Of']
            for word in discard_words:
                gen_words.discard(word)
                next_words.discard(word)
            
            if next_words.issubset(gen_words):
                match_num += 1
        
    return match_num


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

def remove_symbols(text):
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z]')
    res = pattern.sub(' ', text)
    return res


def prompt_assemble(row):
    user_header = ["Gender","Age","Job"]
    user_info = [row["Gender"], row["Age"], row["Job"]]
    user_data = table_to_markdown([user_header, user_info])
    #print("user_data:\n",user_data)
    
    item_header = ["Title","Genres","Rating"]
    item_info = parse_movie_string(row["Hist"])
    
    item_data =[item_header]
    filter_item_info = [item for item in item_info if len(item) == 3 and int(item[2]) <= 3]
    if len(filter_item_info) > 50:
        filter_item_info = filter_item_info[-50:]
                
    for item in filter_item_info:
        item_data.append(item)
    try:
        item_data = table_to_markdown(item_data)
    except:
        item_data = ''
    #print("item_data:\n", item_data)
    prompt = PROMPT.format(user_info = user_data, item_info = item_data)
    
    return prompt
    

def infer_and_match(csv_file_path='/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/llm_data_test_5.csv'):
    parser = argparse.ArgumentParser(description='process model file')
    parser.add_argument('--model', type=str, default="/mnt/data/0/llama3-8b-instruct", help='model file path')

    args = parser.parse_args()

    model_name = args.model

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096, stop='<|eot_id|>')
    llm = LLM(model=model_name)

    with open(csv_file_path,'r',encoding='utf-8') as r_file:
        datas = pd.read_csv(csv_file_path)
        # datas = datas.head(3)
        datas['prompt'] = datas.apply(prompt_assemble, axis=1)
        total_match = 0
        total_mismatch = 0
        prompts = datas['prompt'].tolist()
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        output_texts = [output.outputs[0].text for output in outputs]

        datas['gen_movie'] = pd.Series(output_texts)
        
        datas['match'] = datas.apply(movie_match, axis=1)
        datas['dismatch'] = datas.apply(movie_dismatch, axis=1)
        
        total_match = datas['match'].sum()
        total_mismatch = datas['dismatch'].sum()

    print('total match number is:', total_match, "  total mismatch number is:", total_mismatch)
    return total_match, total_mismatch



if __name__ == '__main__':
    infer_and_match()




