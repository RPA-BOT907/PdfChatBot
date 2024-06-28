### Chat with your PDFs using LangChain and FAISS: A Step-by-Step Guide

In today’s data-driven world, accessing and extracting meaningful information from large documents can be a challenging task. This is especially true when dealing with extensive PDFs that contain a wealth of information. Fortunately, with the advancements in AI and machine learning, there are now sophisticated tools and libraries that can make this task more manageable and efficient.

In this blog, I’ll walk you through a practical example of how to extract insights from a PDF using Python, LangChain, and FAISS. We will specifically look at how to process a PDF document, split the text into manageable chunks, generate embeddings for these chunks, and then utilize these embeddings to answer specific queries about the document. Let's dive in!

#### Tools and Libraries

Before we get started, let’s take a look at the libraries we’ll be using:

- **PyPDF2**: A library for reading PDF files in Python.
- **LangChain**: A library for language model chains, which helps in handling text and generating embeddings.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors, which helps in indexing and searching through embeddings.
- **Ollama**: A model interface for generating embeddings and running question-answering tasks.

#### Step-by-Step Implementation

1. **Read the PDF**

First, we need to read the PDF file and extract its text content. Here’s how you can do it using PyPDF2:

```python
from PyPDF2 import PdfReader

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
```

This code snippet reads the content of each page in the PDF and concatenates it into a single string, `raw_text`.

2. **Split the Text**

To handle large texts efficiently, we split them into smaller chunks. We’ll use the `CharacterTextSplitter` from LangChain for this purpose:

```python
from langchain.text_splitter import CharacterTextSplitter

# Text splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)
```

This splits the text into chunks of 800 characters each, with an overlap of 200 characters to maintain context.

3. **Generate Embeddings**

Next, we generate embeddings for these text chunks using the `OllamaEmbeddings`:

```python
from langchain_community.embeddings import OllamaEmbeddings

# Word Embeddings
ollama_emb = OllamaEmbeddings(
    model="llama2",
)
```

4. **Index the Text Chunks**

We use FAISS to index these embeddings, making it easier to search through them:

```python
from langchain_community.vectorstores import FAISS

# Document search
document_search = FAISS.from_texts(texts, ollama_emb)
```

5. **Create a QA Chain**

To facilitate question answering, we create a QA chain using the `Ollama` model:

```python
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama

# Create QA chain
chain = load_qa_chain(Ollama(), chain_type="stuff")
```

6. **Query the Document**

Finally, we can query the document. Let’s say we want to know about "Jagananna Gorumudda":

```python
query = "what is the jagananna Gorumudda"

docs = document_search.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)

print(answer)
```

This code snippet searches for the text chunks most similar to the query and uses the QA chain to generate an answer based on those chunks.

#### Conclusion

By following these steps, you can efficiently extract and query information from large PDF documents. This approach leverages the power of AI to handle extensive text data, making it a valuable tool for researchers, analysts, and anyone working with large documents.

The integration of PyPDF2, LangChain, FAISS, and Ollama demonstrates how modern libraries can work together to simplify complex tasks. Whether you're dealing with academic papers, business reports, or any other large documents, this method can save you significant time and effort.

Feel free to experiment with the code and adapt it to your specific needs. Happy coding!

---

By following the above guide, you can unlock the potential of your PDF documents, extracting insights and answering queries with ease. This approach not only enhances your productivity but also leverages cutting-edge AI technology to handle data more efficiently.
