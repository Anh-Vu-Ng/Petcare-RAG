from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def splits_documents(documents):
    """
    Chia nhỏ văn bản thành các chunk để đưa vào vector store.
    Sử dụng Markdown Header-based splitting cho URL documents,
    và RecursiveCharacterTextSplitter cho PDF hoặc các tài liệu khác.
    """
    if not documents:
        print("Cảnh báo: Không có document nào để chia nhỏ!")
        return []
        
    pdf_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        # Phân loại tài liệu:
        # Nếu là tài liệu từ URL thì dùng Markdown Splitter
        if source.startswith(("https://")):
            try:
                # 1. Phân tách bằng cấu trúc tiêu đề Markdown
                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on,
                    strip_headers=False  # Giữ tiêu đề lại trong văn bản để LLM có ngữ cảnh hoàn chỉnh
                )
                md_chunks = markdown_splitter.split_text(doc.page_content)
                
                # 2. Với từng chunk tiêu đề, chạy qua pdf_splitter để giới hạn kích thước nếu cần (Bounded)
                for chunk in md_chunks:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update(chunk.metadata)
                    
                    sub_docs = pdf_splitter.split_documents([
                        Document(page_content=chunk.page_content, metadata=chunk_metadata)
                    ])
                    all_chunks.extend(sub_docs)
            except Exception as e:
                print(f"Lỗi khi chia Markdown cho URL {source}: {e}. Fallback sang chia đệ quy thông thường.")
                chunks = pdf_splitter.split_documents([doc])
                all_chunks.extend(chunks)
        else:
            # Tài liệu PDF hoặc các nguồn khác
            chunks = pdf_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            
    # Loại bỏ các chunk trùng lặp nội dung
    unique_chunks = []
    seen_contents = set()
    
    for chunk in all_chunks:
        if chunk.page_content not in seen_contents:
            seen_contents.add(chunk.page_content)
            unique_chunks.append(chunk)
            
    print(f"Đã chia nhỏ dữ liệu thành {len(all_chunks)} chunks. Giữ lại {len(unique_chunks)} chunks duy nhất.")
    return unique_chunks