import os

os.chdir(os.path.dirname(__file__))
input_file_path = 'input.txt'
input_file_source_url = 'https://huggingface.co/datasets/mcapodici/reddit_sydney/resolve/main/input.txt'

if not os.path.exists(input_file_path):
    print("Input file doesn't exist. Downloading.")
    os.system(f'wget -O {input_file_path} {input_file_source_url}')
