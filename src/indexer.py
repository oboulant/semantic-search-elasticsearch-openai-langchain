from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch

from config import Paths, openai_api_key


def main():

    loader = PyPDFLoader(str(Paths.manual))
    documents_manual = loader.load_and_split()

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = ElasticVectorSearch.from_documents(
        documents_manual,
        embeddings,
        elasticsearch_url="http://localhost:9200",
        index_name="elastic-index",
    )
    print(db.client.info())


if __name__ == "__main__":
    main()
