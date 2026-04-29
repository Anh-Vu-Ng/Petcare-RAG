from typing import List
from importlib.metadata import metadata
import fitz #pymupdf
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
from src.config import PDF_FILE, URL_FILE

def load_pdf(pdf_path: str) -> List[Document]:
    """Load pdf with pymupdf"""
    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata = {
                    "source": pdf_path,
                    "page_number": page_num + 1                }
            ))
    return documents

def load_urls(url_file: str) -> List[Document]:
    """Load url from urls.txt"""
    documents =[]
    try: 
        with open(url_file, "r", encoding = "utf-8 ") as f:
            urls = [line.strip() for line in f if line.strip()]

        for url in urls:
            try:
                response = requests.get(url, timeout = 10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                article = soup.find('article')
                if article:
                    for unwanted in article.find_all('div', class_='post-social-tags'):
                        unwanted.decompose()
                    for meta in article.find_all('div', class_='header_meta'):
                        meta.decompose()

                text = article.get_text(separator='\n', strip = True)
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": url
                        }
                    ))
            except Exception as e:
                print(f"Error Loading {url}: {e}")
    except Exception as e:
        print(f"Error reading {url_file}: {e}")
    return documents
def load_all_docs() -> List[Document]:
    pdf_docs = load_pdf(PDF_FILE) if PDF_FILE else []
    url_docs = load_urls(URL_FILE) if URL_FILE else []
    return pdf_docs + url_docs

    

