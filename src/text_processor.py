from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def splits_documents(documents):
    """
    Chia nhỏ văn bản thành các chunk để đưa vào vector store.
    """
    if not documents:
        print("Cảnh báo: Không có document nào để chia nhỏ!")
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Trật tự phân cách tối ưu để giữ toàn vẹn ngữ cảnh của đoạn văn
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Loại bỏ các chunk trùng lặp nội dung (giữ lại 1 bản duy nhất)
    unique_chunks = []
    seen_contents = set()
    
    for chunk in chunks:
        if chunk.page_content not in seen_contents:
            seen_contents.add(chunk.page_content)
            unique_chunks.append(chunk)
            
    print(f"Đã chia nhỏ dữ liệu thành {len(chunks)} chunks. Giữ lại {len(unique_chunks)} chunks duy nhất.")
    return unique_chunks