import uuid
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from src.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP



def split_parent_child(documents):
    """
    Chia tách documents thành parent chunks  và child chunks.
    Trả về:
        parent_docs: dict {parent_id: Document}
        child_docs: List[Document] (mỗi document đều có parent_id trong metadata)
    """
    if not documents:
        print("Cảnh báo: Không có document nào để chia nhỏ!")
        return {}, []

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    parent_docs = {}
    child_docs = []

    for doc in documents:
        source = doc.metadata.get("source", "")
        doc_parents = []
        
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
                
                # 2. Với từng chunk tiêu đề, chia thành các parent chunks
                for chunk in md_chunks:
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update(chunk.metadata)
                    
                    sub_parents = parent_splitter.split_documents([
                        Document(page_content=chunk.page_content, metadata=chunk_metadata)
                    ])
                    doc_parents.extend(sub_parents)
            except Exception as e:
                print(f"Lỗi khi chia Markdown cho URL {source}: {e}. Fallback sang chia đệ quy thông thường.")
                sub_parents = parent_splitter.split_documents([doc])
                doc_parents.extend(sub_parents)
        else:
            # Tài liệu PDF hoặc các nguồn khác
            sub_parents = parent_splitter.split_documents([doc])
            doc_parents.extend(sub_parents)

        # Loại bỏ các parent chunk trùng lặp nội dung trong cùng document
        seen_parent_contents = set()
        unique_parents = []
        for p in doc_parents:
            if p.page_content not in seen_parent_contents:
                seen_parent_contents.add(p.page_content)
                unique_parents.append(p)

        # Với mỗi parent chunk, tạo một parent_id, lưu lại và tiếp tục chia nhỏ thành child chunks
        for p in unique_parents:
            parent_id = str(uuid.uuid4())
            parent_docs[parent_id] = p
            
            # Chia nhỏ parent chunk thành các child chunks
            sub_children = child_splitter.split_documents([p])
            for c in sub_children:
                c.metadata["parent_id"] = parent_id
                child_docs.append(c)

    # Loại bỏ các child chunk trùng lặp nội dung
    unique_children = []
    seen_child_contents = set()
    for c in child_docs:
        if c.page_content not in seen_child_contents:
            seen_child_contents.add(c.page_content)
            unique_children.append(c)

    print(f"Đã chia dữ liệu thành {len(parent_docs)} parent chunks và {len(unique_children)} child chunks.")
    return parent_docs, unique_children