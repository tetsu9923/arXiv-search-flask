import gc
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, render_template, request

from memory_profiler import profile


@profile
def embed(raw_text):
    #tokenizer = AutoTokenizer.from_pretrained('./model')
    #model = AutoModel.from_pretrained('./model', low_cpu_mem_usage=True).to('cpu')
    with torch.no_grad():
        _input = tokenizer(raw_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to('cpu')
        output = model(**_input).pooler_output.detach().numpy()
    return output

def cos_similarity(x, y, eps=1e-16):
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny) 

def get_rawtext(idx):
    with open('./data/raw_link.pkl', 'rb') as f:
        link_list = pickle.load(f)
    link = link_list[idx]
    for i in range(n_split):
        if idx < (len(link_list)//n_split)*(i+1):
            with open('./data/raw_title/raw_title{}.pkl'.format(i+1), 'rb') as f:
                raw_title = pickle.load(f)
            with open('./data/raw_abst/raw_abst{}.pkl'.format(i+1), 'rb') as f:
                raw_abst = pickle.load(f)

            title = raw_title[idx-(len(link_list)//n_split)*i]
            abst = raw_abst[idx-(len(link_list)//n_split)*i]
            break
    return title, abst, link

@profile
def check_memory():
    return


app = Flask(__name__)
n_split = 30

tokenizer = AutoTokenizer.from_pretrained('./model')
model = AutoModel.from_pretrained('./model', low_cpu_mem_usage=True).to('cpu')

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        title = request.form['title']
        abst = request.form['abstract']

        use_title = use_abst = False

        if len(title) > 0:
            use_title = True
            query_title = embed(title)
        if len(abst) > 0:
            use_abst = True
            query_abst = embed(abst)

        if use_title and use_abst:
            query = np.concatenate([query_title, query_abst], axis=1)
        elif use_title:
            query = query_title.copy()
        elif use_abst:
            query = query_abst.copy()
        else:
            raise ValueError('Enter either title or abstract and try again.')

        sim_list = []
        for i in range(n_split):
            database_title = np.load('./data/database_title/database_title{}.npy'.format(i+1))
            database_abst = np.load('./data/database_abst/database_abst{}.npy'.format(i+1))

            if use_title and use_abst:
                database = np.concatenate([database_title, database_abst], axis=1)
            elif use_title:
                database = database_title.copy()
            elif use_abst:
                database = database_abst.copy()
            del database_title
            del database_abst

            for vector in database:
                sim_list.append(cos_similarity(query, vector).item())
            del database
            gc.collect()

        sim_list = np.array(sim_list)
        sim_idx = np.argsort(sim_list)[::-1]
        title, abst, link = get_rawtext(sim_idx[0].item())
        return render_template('index.html', line1=title, line2=abst, line3=link)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)