from typing import List, Optional
from importlib.metadata import metadata
import re
import fitz #pymupdf
from bs4 import BeautifulSoup
import requests
from langchain_core.documents import Document
from src.config import PDF_FILE, URL_FILE

def clean_text(text: str) -> str:
    # Thay thế nhiều khoảng trắng liên tiếp (spaces, tabs) thành 1 space
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Loại bỏ khoảng trắng đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.splitlines()]
    # Loại bỏ các dòng trống, nối lại bằng 1 dấu xuống dòng
    text = '\n'.join(line for line in lines if line)
    return text.strip()

def load_pdf(pdf_path: str) -> List[Document]:
    """Load pdf with pymupdf"""
    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        text = clean_text(page.get_text())
        if text:
            documents.append(Document(
                page_content=text,
                metadata = {
                    "source": pdf_path,
                    "page_number": page_num + 1                }
            ))
    return documents

def load_single_url(url: str) -> Optional[Document]:
    """Tải và parse nội dung của một URL đơn lẻ."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Thử tìm các tag chính chứa nội dung
        article = soup.find('article') or soup.find('main')
        
        if not article:
            # Fallback: lấy thẻ body và loại bỏ các thành phần gây nhiễu
            article = soup.body
            if article:
                # Loại bỏ các thẻ rác để lấy text chính xác hơn
                for tag in article.find_all(['nav', 'header', 'footer', 'script', 'style', 'iframe', 'noscript']):
                    tag.decompose()
        
        final_text = ""
        if article:
            content_tags = article.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li'])
            raw_text_chunks = []
            for tag in content_tags:
                if tag.find_parent(['p', 'blockquote', 'li']):
                    continue
                text = tag.get_text(separator=" ", strip=True)
                if text:
                    raw_text_chunks.append(text)
                    
            if raw_text_chunks:
                # Gộp chuỗi ngoài vòng lặp để tối ưu hóa hiệu năng O(N)
                final_text = "\n".join(raw_text_chunks)
                final_text = re.sub(r' +', ' ', final_text)
        
        if final_text:
            return Document(
                page_content=final_text,
                metadata={
                    "source": url
                }
            )
    except Exception as e:
        print(f"Error Loading {url}: {e}")
    return None

def load_urls(url_file: str) -> List[Document]:
    """Tải nội dung từ danh sách các URL song song bằng ThreadPoolExecutor."""
    documents = []
    try:
        with open(url_file, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
            
        if not urls:
            return []
            
        import concurrent.futures
        # Sử dụng tối đa 5 workers song song để crawl nhanh hơn
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # map trả về kết quả theo đúng thứ tự các URL
            results = executor.map(load_single_url, urls)
            for doc in results:
                if doc:
                    documents.append(doc)
    except Exception as e:
        print(f"Error reading {url_file}: {e}")
    return documents
def load_all_docs() -> List[Document]:
    pdf_docs = load_pdf(PDF_FILE) if PDF_FILE else []
    url_docs = load_urls(URL_FILE) if URL_FILE else []
    return pdf_docs + url_docs

    

