import os
import re
import tqdm

import pandas as pd

from langchain_community.document_loaders import PyPDFLoader

conference = 'acl'

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
            abstract = 'Abstract'+header[:match.start()]
        else:
            abstract = ""
    else:
        abstract = ""
    
    return abstract

def extract_references(text):
    references = ""
    pattern = re.compile(r'\sReference[s]?', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        references = text[match.start():]
    else:
        references = ""
    return references



for year in range(2015, 2025):
    str_path = "ACL/acl"
    print(f"Starting {year} data conversion")
    data = pd.DataFrame(columns=['title', 'header', 'abstract', 'main_body', 'reference', 'text', 'year', 'pages', 'conference'])

    lst_data = []
    lst_pdfs = [f for f in os.listdir(f'./data/{str_path}_{year}_main') if f.endswith('.pdf')]

    for idx, path in enumerate(lst_pdfs):
        print(f"Idx: {idx}\tFile: {path}")
        # Read pdf
        # pdf = PyPDFLoader(f'./data/{str_path}_{year}_main/{path}')

        pdf = PyPDFLoader(path)

        try:
            text = pdf.load()
        except Exception as e:
            print("Error: ", e)
            print(f"Idx: {idx} and name {path}")
            continue

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

        # Extract references
        references = extract_references(page_content)

        # Extract main body (without abstract and references)
        main_body = page_content
        if abstract != "":
            main_body = main_body.replace(abstract, "")
        if references != "":
            main_body = main_body.replace(references, "")


        row = {'titel': titel, 'header': header, 'abstract':abstract, 'main_body': main_body, 'references': references, 'text': page_content, 'year': year, 'pages': total_pages, 'conference': conference}
        lst_data.append(row)

    data = pd.DataFrame(lst_data)
    data.to_pickle(f'./data/{str_path}_{year}_main.pkl')

    print(f"Finished writing {year} data to csv")