import requests
from bs4 import BeautifulSoup

def xml_parser(url):
    rows = []
    response = requests.get(url, verify=False)
    xml_content = response.content
    soup = BeautifulSoup(xml_content, features="xml")
    texts = str(soup.find_all(string=True)).replace('\\n','')
    child = soup.find("item")
    while True:    
            row = {'ticker': 'NFLX', 'company_name': 'Netflix'}
            try:
                row['annoucement_date'] = (" ".join(child.find('pubDate')))
            except:
                row['annoucement_date'] = " "
                
            try:
                row['use_case'] = (" ".join(child.find('title')))
            except:
                row['use_case'] = " "
            
            try:   
                rows.append(row)
                child = child.find_next_sibling('item')
            except:
                break
    print("XML Parser Complete")
    return rows
