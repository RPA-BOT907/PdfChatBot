from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama

# Define the path to the PDF file
pdf_path = "C:\\Users\\haris\\OneDrive\\Desktop\\Child-Budget_removed.pdf"

# Initialize the PDF reader
pdf_file = PdfReader(pdf_path)

# Read text from PDF
raw_text = ''
for i, page in enumerate(pdf_file.pages):
    content = page.extract_text()
    if content:
        raw_text += content

#text splitting
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)

texts = text_splitter.split_text(raw_text)

# Print the extracted text
#print(texts)
#print(len(texts))


#wordEmbeddings
ollama_emb = OllamaEmbeddings(
    model="llama2",
)

#Document search
document_search = FAISS.from_texts(texts, ollama_emb)


#print(document_search)


#create chain
chain = load_qa_chain(Ollama(), chain_type="stuff")

query = "what is the jagananna Gorumudda"
docs = document_search.similarity_search(query)
answer=chain.run(input_documents=docs, question=query)

print(answer)