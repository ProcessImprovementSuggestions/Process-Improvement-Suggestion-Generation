import grobid_tei_xml
import numpy as np
import requests
from itertools import chain
from langchain.document_loaders import WebBaseLoader
from processing_utils import resource_preprocessing
from processing_utils import vector_db
from protego import Protego
from urllib import parse
from urllib.parse import urlsplit


def get_tweet_documents(query, qdrantdb_client, tweet_collection_name, embedder, limit_results):
    tweet_search_results = vector_db.search_kb(query, qdrantdb_client, tweet_collection_name, embedder, limit_results)
    return [[tweet_search_result['page_content'], tweet_search_result['source'], 'tweet_id'] for tweet_search_result in tweet_search_results]


def crawl_allowed(full_url):
    #check whether robots.txt allows us to crawl website
    url_parts = urlsplit(full_url)
    base_url = url_parts.scheme + "://" + url_parts.netloc
    try:
        r = requests.get(parse.urljoin(base_url, 'robots.txt'), timeout=5)
        rp = Protego.parse(r.text)
        is_allowed = rp.can_fetch(full_url, "*")
    except:
        is_allowed = False
    return is_allowed


#Semantic_scholar
def request_papers(ids, fields, x_api_key):
    if len(ids) > 500:
        requested_papers = False
        print("Too many ids")
    else:
        try:
            response = requests.post(
                'https://api.semanticscholar.org/graph/v1/paper/batch',
                params = {'fields': fields},
                headers = {'x-api-key': x_api_key},
                json = {"ids": ids}
            )
            response.raise_for_status()
            requested_papers = response.json()
        except:
            requested_papers = False
        
    return requested_papers


def format_context(context):
    context = "\n\n".join(context)
    return context


def get_paper_documents(query, qdrantdb_client, tldr_collection_name, embedder, url_setting, x_api_key, limit_results):
    tldr_search_results = vector_db.search_kb(query, qdrantdb_client, tldr_collection_name, embedder, limit_results)
    tldr_search_results = [[tldr_search_result['page_content'], tldr_search_result['source'], 'corpus_id'] for tldr_search_result in tldr_search_results]

    #Loading and parsing PDFs
    n_a_papers = []
    fields = 'tldr,openAccessPdf,title,corpusId,isOpenAccess'

    paper_sources = [tldr_search_result[1] for tldr_search_result in tldr_search_results]
    request_paper_ids = [f'CorpusId:{str(source_id)}' for source_id in paper_sources]

    requested_papers = request_papers(request_paper_ids, fields, x_api_key)

    if requested_papers != False:
        for paper_i in requested_papers:
            if paper_i == None:
                continue
            if ((paper_i["isOpenAccess"] == True) and (paper_i["openAccessPdf"] != None)):
                try:
                    paper_url = paper_i["openAccessPdf"]["url"]
                    if (crawl_allowed(paper_url) == True):
                        pdf_resp = requests.get(paper_url, allow_redirects=True)
                        xml = requests.post(url_setting, files={'input': pdf_resp.content})
                        doc = grobid_tei_xml.parse_document_xml(xml.text)
                        #print(f'Open Access Paper: {paper_url}')
                        if ((doc.body != None) and (doc.abstract != None)):
                            for i, tldr_search_result_i in enumerate(tldr_search_results):
                                if str(tldr_search_result_i[1]) == str(paper_i["corpusId"]):
                                    tldr_search_results[i][0] = format_context([doc.abstract, doc.body])
                        elif ((doc.body != None)):
                            for i, tldr_search_result_i in enumerate(tldr_search_results):
                                if str(tldr_search_result_i[1]) == str(paper_i["corpusId"]):
                                    tldr_search_results[i][0] = doc.body
                        elif ((doc.abstract != None)):
                            for i, tldr_search_result_i in enumerate(tldr_search_results):
                                if str(tldr_search_result_i[1]) == str(paper_i["corpusId"]):
                                    tldr_search_results[i][0] = doc.abstract                                    
                        else:
                            n_a_papers.append(paper_i["corpusId"])
                    else:
                        n_a_papers.append(paper_i["corpusId"])
                except:
                    n_a_papers.append(paper_i["corpusId"])                
            else:
                n_a_papers.append(paper_i["corpusId"])


    #Generate suggestions
    text_results = [tldr_search_result[0] for tldr_search_result in tldr_search_results]
    metadata_results = [{'source': tldr_search_result[1]} for tldr_search_result in tldr_search_results]
    splits = resource_preprocessing.create_split_documents(text_results, metadata_results, embedder)
    splits = [[split['page_content'], split['metadata']['source'], 'corpus_id'] for split in splits]

    return splits


def search_web(query, service, GOOGLE_CSE_ID, limit_results):
    responses = []
    links = []

    query = f"{query} -filetype:pdf"

    for start_i in range(1,limit_results+1,10):
        res = service.cse().list(
            q=query,
            cx=GOOGLE_CSE_ID,
            num=min(10, limit_results-start_i+1), #Valid values are integers between 1 and 10, inclusive.
            start=start_i
        ).execute()
        responses.extend(res['items'])

    for result in responses:
        links.append(result['link'])

    return links


def get_web_documents(query, service, GOOGLE_CSE_ID, embedder, limit_results):
    web_results = search_web(query, service, GOOGLE_CSE_ID, limit_results)

    n_a_websites = []
    allowed_web_results = []
    for result_i in web_results:
        if crawl_allowed(result_i) == True:
            allowed_web_results.append(result_i)
        else:
            n_a_websites.append(result_i)

    #Load websites
    docs = []
    for allowed_web_result_i in allowed_web_results:
        try:
            loader = WebBaseLoader(allowed_web_result_i, requests_kwargs={'timeout':5})
            docs.append(loader.load()[0])
        except:
            n_a_websites.append(allowed_web_result_i)

    #Embed websites
    web_texts = [web_document.page_content for web_document in docs]
    web_metadata = [web_document.metadata for web_document in docs]
    web_documents = resource_preprocessing.create_split_documents(web_texts, web_metadata, embedder)

    return [[web_document['page_content'], web_document['metadata']['source'], 'web_link'] for web_document in web_documents]

