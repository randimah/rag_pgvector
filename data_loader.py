import datasets
from typing import  List
from langchain_core.documents import Document


def load_data() -> List[Document]:
    """
    Loads the document dataset from hugging face
    :return: langchain document list
    """
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
    knowledge_base = [
        Document(
            page_content=doc["text"],
            metadata={"source": doc["source"]}) for doc in ds
    ]
    return knowledge_base