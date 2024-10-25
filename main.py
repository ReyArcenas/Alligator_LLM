from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from faiss import write_index,  read_index
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
# os.environ["OPENAI_API_KEY"] = getpass.getpass()

llm = ChatOpenAI(model="gpt-4o-mini")

loader = DirectoryLoader(path="docs")

documents = loader.load()

template = """ Write an article similar to these UF Alligator Articles provided. Note that some articles are poorly scanned OCR and may have some gibberish. 
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

#db = FAISS.from_documents(texts, embeddings)
db = FAISS.load_local('persist', embeddings, allow_dangerous_deserialization=True)
db.save_local('persist')

retriever = db.as_retriever()
#docs = retriever.invoke("What did they say about the florida mens basketball team?")

#print(docs)


def format_docs(contents):
    return "\n\n".join([d.page_content for d in contents])


chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print(chain.invoke("Make a short 100-word maximum Alligator article about a recent UF alum."))
