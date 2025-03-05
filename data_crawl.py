# %%
from bs4 import BeautifulSoup
import json
import numpy as np
import requests
import os
from tqdm import tqdm

page_url = "https://aclanthology.org/events/acl-2023/"
conf_name = './data/ACL/acl_2023_main'
conf_list = ['2023acl-long', '2023acl-short']

#%%
html_doc = requests.get(page_url).text
soup = BeautifulSoup(html_doc, 'html.parser')
# %%
main_papers = []
for conf_id in conf_list:
    main_papers += soup.find('div', id = conf_id).find_all('p', class_ = "d-sm-flex")

# %%
paper_list = []
print(f"Found {len(main_papers)} papers")
input("continue?")
for paper_p in main_papers:
    pdf_url = paper_p.contents[0].contents[0]['href']
    paper_span = paper_p.contents[-1]
    assert paper_span.name == 'span'
    paper_a = paper_span.strong.a
    title = paper_a.get_text()
    url = "https://aclanthology.org" + paper_a['href']
    paper_list.append([title, url, pdf_url])

# %%
with open(conf_name + '.json', 'w', encoding='utf8') as f:
    json.dump(paper_list, f, indent = 2, ensure_ascii= False)

print('There are total {} papers'.format(len(paper_list)))

if not os.path.exists(conf_name):
    os.mkdir(conf_name)

illegal_chr = r'\/:*?<>|'
table = ''.maketrans('', '', illegal_chr)
for i, paper in tqdm(list(enumerate(paper_list))):
    try:
        r = requests.get(paper[2])
        n = '{}.{}.pdf'.format(i+1, paper[0].translate(table))
        with open('./{}/{}'.format(conf_name, n), 'wb') as f:
            f.write(r.content)
    except Exception as e:
        print("Error: ", e)
        continue
