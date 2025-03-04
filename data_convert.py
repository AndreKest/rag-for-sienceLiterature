import os
import re
import tqdm

import pandas as pd

from langchain.document_loaders import PyPDFLoader

year = 2023
data = pd.DataFrame(columns=['title', 'text', 'year', 'pages'])

lst_data = []
lst_pdfs = [f for f in os.listdir(f'./data/EMNLP/emnlp_{year}_main') if f.endswith('.pdf')]

for idx, path in enumerate(lst_pdfs):
    print(f"Idx: {idx}\tFile: {path}")
    # Read pdf
    pdf = PyPDFLoader(f'./data/EMNLP/emnlp_{year}_main/{path}')
    text = pdf.load()

    # Extract title
    titel = text[0].metadata['source']
    total_pages = text[0].metadata['total_pages']

    # Extract title between number*. and .pdf
    match = re.search(r'\d+\.(.*)\.pdf', titel)
    if match:
        titel = match.group(1)
    
    page_content = ""
    for t in text:
        page_content += " " + t.page_content

    row = {'titel': titel, 'text': page_content, 'year': year, 'pages': total_pages}
    lst_data.append(row)

data = pd.DataFrame(lst_data)
data.to_csv(f'./data/EMNLP/emnlp_{year}_main.csv', index=False, encoding='utf-8', errors='ignore')