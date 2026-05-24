import os
import sys

# Force stdout/stderr to use UTF-8 to prevent CP1252 encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add workspace root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.text_processor import splits_documents

# Mock URL Document formatted as Markdown
mock_url_doc = Document(
    page_content=(
        "# Daily Pet Care\n\n"
        "Welcome to the dog and cat care manual.\n\n"
        "## Diet and Nutrition\n\n"
        "Nutrition is key to health.\n"
        "Dogs need animal protein.\n"
        "Cats need taurine for bright eyes and a healthy heart.\n\n"
        "### Foods to Avoid\n\n"
        "Do not feed chocolate to dogs because it contains theobromine.\n"
        "Avoid raisins, onions, and garlic.\n\n"
        "## Vaccination Schedule\n\n"
        "Vaccinations help prevent dangerous infectious diseases."
    ),
    metadata={"source": "https://petcare.example.com/guide"}
)

# Mock PDF Document (Normal)
mock_pdf_doc = Document(
    page_content=(
        "Technical guide for veterinary care.\n"
        "This section explains basic surgery procedures.\n"
        "Use general anesthesia for surgery."
    ),
    metadata={"source": "data/sample_guide.pdf"}
)

def main():
    print("--- Running splits_documents test ---")
    documents = [mock_url_doc, mock_pdf_doc]
    chunks = splits_documents(documents)
    
    print(f"\nSuccessfully split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i+1}] Source: {chunk.metadata.get('source')}")
        # Filter and print headers in metadata
        headers = {k: v for k, v in chunk.metadata.items() if k.startswith("Header")}
        if headers:
            print(f"Headers: {headers}")
        print("Content:")
        print(chunk.page_content)
        print("-" * 50)

if __name__ == "__main__":
    main()
