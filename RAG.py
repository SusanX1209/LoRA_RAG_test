
import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from transformers import pipeline
import transformers
from langchain_community.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader,DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./log/rag.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)

class Llama3_8B_gen:
    def __init__(self, pipeline, terminators):
        self.pipeline = pipeline
        self.terminators = terminators
        
    def generate_prompt(self, query, retrieved_text):
        messages = [
            {"role": "system", "content": "Answer the Question for the Given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
    
    def generate(self,query,retrieved_context):
        prompt = self.generate_prompt(query ,retrieved_context)
        output =  self.pipeline(prompt,max_new_tokens=512,eos_token_id=self.terminators,do_sample=True,
                            temperature=0.7,top_p=0.9,)   
        # logger.info(f"output from pipeline:{output}")      
        return output[0]["generated_text"][len(prompt):]

class langchain_community_RAG:
    def __init__(self, rag_file_dir, index_path):
        # self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.index_path = index_path
        self.rag_file_dir = rag_file_dir

        if os.path.exists(index_path):
            logger.info("<< Loading FAISS index from cache")
            self.get_vec_value = FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
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

            logger.info(f"<< Generating FAISS index with {len(chunks)} chunks")
            self.get_vec_value = FAISS.from_documents(chunks, self.embeddings)
            self.get_vec_value.save_local(index_path)

        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4})

    def __call__(self, query):
        results = self.retriever.get_relevant_documents(query)
        # logger.info(f"direct retriever content: {results}")
        return "".join([doc.page_content for doc in results])


def main():

    os.environ["HF_TOKEN"]=""

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        token=os.getenv("HF_TOKEN"),
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": quant_config,
            "low_cpu_mem_usage": True,
        },
    )

    terminators =  [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    rag_file_dir = "./data"
    FAISS_index_cache = "faiss_index_cache_1000"


    text_gen = Llama3_8B_gen(pipeline=pipeline, terminators=terminators)
    retriever = langchain_community_RAG(rag_file_dir=rag_file_dir, index_path=FAISS_index_cache)

    def Rag_qa(query):
        retriever_context = retriever(query)
        # logger.info(f"retriever_context:{retriever_context}")
        result = text_gen.generate(query,retriever_context)
        return result

    result1 = Rag_qa("萧炎认识哪些女性？")
    result2 = Rag_qa("萧炎有哪些成长节点？")
    result3 = Rag_qa("萧炎有哪些武器和技能？")
    result4 = Rag_qa("陆阳有哪些武器和技能？")

    with open("./result/doupo.txt", 'w', encoding='utf-8') as f:
        f.write("问题1：萧炎认识哪些女性？\n")
        f.write(result1 + "\n\n")

        f.write("问题2：萧炎有哪些成长节点？\n")
        f.write(result2 + "\n\n")

        f.write("问题3：萧炎有哪些武器和技能？\n")
        f.write(result3 + "\n\n")

        f.write("问题4：陆阳有哪些武器和技能？\n")
        f.write(result4 + "\n\n")



if __name__ == "__main__":

    main()


