import os

os.chdir(os.path.dirname(__file__))
input_file_path = 'input.txt'
file_name_zipped = 'reddit_sydney_text_sample.tgz'
data_url_zipped = 'https://q1r1.c19.e2-5.dev/models/reddit_sydney_text_sample.tgz'

if not os.path.exists(file_name_zipped):
    print("Input file doesn't exist. Downloading and extracting.")
    os.system(f'wget -O --no-check-certificate {file_name_zipped} {data_url_zipped}')

if not os.path.exists(input_file_path):
    print("Extracting...")
    os.system(f'tar -xf {file_name_zipped}')