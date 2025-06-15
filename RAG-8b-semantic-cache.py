import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import transformers
import time
from langchain.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
 

from langchain_community.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

import os
import logging

import pickle
import hashlib

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./log/rag_semantic_cache.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)


### this class used to retrieve the text from pdf and chunk it 
class Langchain_RAG:
    def __init__(self, rag_file_dir, FAISS_index_cache_path, chunks_cache_path):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.rag_file_dir = rag_file_dir
        self.FAISS_index_cache_path = FAISS_index_cache_path
        self.chunks_cache_path = chunks_cache_path
        self.data_hash = compute_data_hash(rag_file_dir)
        if os.path.exists(FAISS_index_cache_path)and is_cache_valid(FAISS_index_cache_path, self.data_hash):
            logger.info("<< Loading FAISS index from cache")
            self.get_vec_value = FAISS.load_local(FAISS_index_cache_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if os.path.exists(chunks_cache_path) and is_cache_valid(chunks_cache_path, self.data_hash):
                logger.info("<< Loading cached chunks from disk")
                with open(chunks_cache_path, 'rb') as f:
                    chunks = pickle.load(f)
            else:
                logger.info("<< Loading and chunking documents...")
                loader_mapping = {
                    ".pdf": PDFMinerLoader,
                    ".txt": TextLoader
                }
                docs = []
                for file in os.listdir(rag_file_dir):
                    ext = os.path.splitext(file)[-1].lower()
                    loader_cls = loader_mapping.get(ext)
                    if loader_cls:
                        loader = loader_cls(os.path.join(rag_file_dir, file))
                        docs.extend(loader.load())
                        
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

                with open(chunks_cache_path, 'wb') as f:
                    pickle.dump(chunks,f)

            logger.info(f"<< Generating FAISS index with {len(chunks)} chunks")
            self.get_vec_value = FAISS.from_documents(chunks, self.embeddings)
            self.get_vec_value.save_local(FAISS_index_cache_path)
            save_cache_meta(FAISS_index_cache_path, self.data_hash)
        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4})

    def __call__(self, query):
        relevant_docs = self.retriever.get_relevant_documents(query)
        return "".join([doc.page_content for doc in relevant_docs])


# This class is used to generate responses from an LLM model
class Llama3_8B_gen:
    def __init__(self, pipeline, embeddings, vector_store, threshold):
        self.pipeline = pipeline
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.threshold = threshold
        
    # @staticmethod
    def generate_prompt(self, query,retrieved_text):
        messages = [
            {"role": "system", "content": "Answer the Question for the Given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    
    def semantic_cache(self, query, prompt):
        query_embedding = self.embeddings.embed_documents([query])
        similar_docs = self.vector_store.similarity_search_with_score_by_vector(query_embedding[0], k=1)

        if similar_docs and similar_docs[0][1] < self.threshold:
            self.print_bold_underline("---->> From Cache")
            return similar_docs[0][0].metadata['response']
        else:
            self.print_bold_underline("---->> From LLM")
            output = self.pipeline(prompt, max_new_tokens=512, eos_token_id=terminators, do_sample=True, temperature=0.7, top_p=0.9)
            
            response = output[0]["generated_text"][len(prompt):]
            self.vector_store.add_texts(texts = [query], 
                       metadatas = [{'response': response},])
            
            # doc = Document(page_content=query, metadata={"response": response})
            # self.vector_store.add_documents([doc])
            return response
            
    def generate(self, query, retrieved_context):
        start_time = time.time()
        
        prompt = self.generate_prompt(query, retrieved_context)
        res = self.semantic_cache(query, prompt)   
        
        end_time = time.time()
        execution_time = end_time - start_time
        self.print_bold_underline(f"LLM generated in {execution_time:.6f} seconds")
        
        return res

    @staticmethod
    def print_bold_underline(text):
        print(f"\033[1m\033[4m{text}\033[0m")


def compute_data_hash(folder_path: str) -> str:
    hash_md5 = hashlib.md5()
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath):
            # 文件名 + 修改时间作为 hash 依据
            hash_md5.update(fname.encode())
            hash_md5.update(str(os.path.getmtime(fpath)).encode())
    return hash_md5.hexdigest()

def save_cache_meta(base_path, data_hash):
    with open(base_path + '.meta', 'w') as f:
        f.write(data_hash)

def is_cache_valid(base_path, current_hash):
    meta_path = base_path + '.meta'
    if not os.path.exists(base_path) or not os.path.exists(meta_path):
        return False
    with open(meta_path, 'r') as f:
        saved_hash = f.read()
    return saved_hash == current_hash


if __name__ ==  '__main__':
    
    FAISS_index_cache_path = 'faiss_index_cache_1000_semantic'
    chunks_cache_path = 'chunks_cache_semantic'
    rag_file_dir = "./data"

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    os.environ["HF_TOKEN"]=""

    rag_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        token=os.getenv("HF_TOKEN"),
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": True},
            "low_cpu_mem_usage": True,
        },
    )

    terminators =  [
        rag_pipeline.tokenizer.eos_token_id,
        rag_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    ### for semantic cache
    # vector_store = FAISS()

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    # Initialize an empty FAISS index
    dimension = embeddings.client.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    docstore = InMemoryDocstore()

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id={}
    )


    text_gen = Llama3_8B_gen(pipeline=rag_pipeline,embeddings=embeddings,
                            vector_store=vector_store,threshold=0.1)
    
    retriever = Langchain_RAG(rag_file_dir=rag_file_dir, FAISS_index_cache_path=FAISS_index_cache_path, chunks_cache_path=chunks_cache_path)

    def Rag_qa(query):
        retriever_context = retriever(query)
        result = text_gen.generate(query,retriever_context)
        return result

    query = ["What is Deep learning ?",
             "萧炎认识哪些女性？",
             "萧炎有哪些成长节点？",
             "萧炎有哪些武器？",
             "萧炎有哪些技能？",
             "陆阳有哪些武器和技能？",
             "萧炎有哪些女性朋友？",
             "萧炎成长路上帮助过他的人有哪些？",
             ]
    response = []
    for q in query:
        response.append(Rag_qa(q))
    
    with open('rag_semantic.txt','w',encoding='utf-8') as f:
        for q,res in zip(query,response):
            f.write(q + '\n' + res + '\n\n')

