import os
import re

from typing import List, Dict

DOCS_DIR = "/documents/website/content/en/docs"

def clean_hugo_md(content: str) -> tuple[str, str]:
    """
    Cleans Hugo markdown content by removing the front matter metadata .

    Args:
        content (str): The raw markdown content.
    Returns:
        tuple[str, str]: A tuple containing the cleaned markdown content and the extracted title.
    """ 
    title =" Kubernetes Documentation" # a fallback when a file doesn't have a specific title

    title_match = re.search(r'^title:\s*["\']?(.*?)["\']?$', content,re.MULTILINE)
    if title_match:
        title = title_match.group(1)
    content = re.sub(r'^---.*?---','' ,content, flags=re.DOTALL) #for the fornt matter
    content = re.sub(r'\{\{[%<].*?[%>]\}\}','',content) #for shortcodes 

    return content.strip(), title

def chunking ( content: str ,chunk_size: int = 800 ,overlap: int = 100) -> List[str]:
    """
    Splits the input text into chunks of specified size with overlap.
    """
    chunks=[]
    start = 0
    while start <len(content):
        end = start +chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        start+= chunk_size - overlap
    return chunks

def ingest_docs(docs_dir: str) -> List[Dict[str, str]]:

    processed_docs = []

    for root, _,files in os.walk(docs_dir):
        for file in files :
            if file.endswith("md"):
                file_path = os.path.join(root,file)
                with open(file_path,"r", encoding="utf-8") as fich :
                    raw_content = fich.read()
                cleaned_content , title = clean_hugo_md(raw_content)
                text_chunk = chunking(cleaned_content)

                for i, chunk in enumerate(text_chunk):
                    processed_docs.append({
                        "text": chunk,
                        "metadata": {
                            "title": title,
                            "source": file_path,
                            "chunk_index": i
                        }
                    })

    return processed_docs

if __name__ == "__main__":
    # Test our loader on a small scale
    print("Loading and chunking Kubernetes documents...")
    # chunks = load_and_process_docs(DOCS_DIR)
    # print(f"Total chunks created: {len(chunks)}")
                
    