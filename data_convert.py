import os
import re
import tqdm

import pandas as pd

from langchain_community.document_loaders import PyPDFLoader

def extract_header(page):
    header = ""

    # Gett all till Abstract
    pattern = re.compile(r'Abstract', re.IGNORECASE)
    match = pattern.search(page.page_content)
    if match:
        header = page.page_content[:match.start()]
    else:
        header = ""


    return header


def extract_abstract(page):
    abstract = ""

    # Gett all between Abstract and 1 Introduction
    pattern = re.compile(r'Abstract', re.IGNORECASE)
    match = pattern.search(page.page_content)
    if match:
        header = page.page_content[match.end():]
        pattern = re.compile(r'1\s*Introduction', re.IGNORECASE)
        match = pattern.search(header)
        if match:
            abstract = header[:match.start()]
        else:
            abstract = ""
    else:
        abstract = ""
    
    return abstract


for year in range(2010, 2013):
    # year = 2021
    print(f"Starting {year} data conversion")
    data = pd.DataFrame(columns=['title', 'header', 'abstract', 'text', 'year', 'pages'])

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

        # extract header
        header = ""
        header = extract_header(page=text[0])

        # extract abstract
        abstract = ""
        abstract = extract_abstract(page=text[0])


        page_content = ""
        for page_num, t in enumerate(text):
            if header != "" and page_num == 0:
                # Remove header from page content
                t.page_content = t.page_content.replace(header, "")

            page_content += " " + t.page_content

        row = {'titel': titel, 'header': header, 'abstract':abstract, 'text': page_content, 'year': year, 'pages': total_pages}
        lst_data.append(row)

    data = pd.DataFrame(lst_data)
    data.to_csv(f'./data/EMNLP/emnlp_{year}_main.csv', index=False, encoding='utf-8', errors='ignore')

    print(f"Finished writing {year} data to csv")