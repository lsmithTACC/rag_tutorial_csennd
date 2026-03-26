# Import torch for general functionality
import torch

# Import tokenizers and model loader for LLM backend
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import langchain tools for RAG
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 

# User controls:
EMBEDDING_MODEL = "./models/Qwen3-Embedding-0.6B"
CHUNK_SIZE = 4096 		# Size of text chunks, in number of characters
CHUNK_OVERLAP = 128		# Size of overlap between, in number of characters
NUM_K = 2 				# Number of nearest neighbors returned in semantic search
MAX_NEW_TOKENS = 64     # Max length of LLM response, in tokens

# Main RAG script with example
def main(model_path):

    # Device detection
    if torch.cuda.is_available():
        device='cuda'
    elif torch.backends.mps.is_available():
        device='mps'
    else:
        device='cpu'

    # ---------- LLM Backend ---------- #

    # Load the tokenizer associated with specific LLM
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load LLM checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    # Update model config
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))


    # ---------- RAG ---------- #

    # Setup document loader
    loader = DirectoryLoader('./docs/', glob="**/*.pdf", show_progress=False, loader_cls=PyPDFLoader, load_hidden=False)
    docs = loader.load()

    # Split text based on specified chunk config
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)

    # Setup database Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # only need to insert document once
    db = Chroma.from_documents(texts, embeddings, persist_directory="db_ce")
    # after you run the code for the first time, you can re-use the databse with the following command
    # db = Chroma(persist_directory="db_ce", embedding_function=embeddings)


    # ---------- Test Case ---------- #
       
    # Test questions
    messages = ["Desribe Aurora's neural network architecture.",
    			"What did the authors use as the training objective?"
				]

    # Loop through test questions
    for question in messages:

    	# Retrieve context from DB
        results = db.similarity_search(question, k=NUM_K)
        retrieved_context = "\n\n".join(result.page_content for result in results)

        # Tokenize prompt and evalulate LLM
        prompt = f"system: You are a helpful AI assistant. \
        Use the following retrieved context to help answer the question. \
        Keep your answers concise. Two sentences maximum. \
        \n Question:{question} \
        \n Retrieved Context: {retrieved_context} \
        \n Answer: "
        inputs = tokenizer(prompt,return_tensors="pt").to(device)
        input_length = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, do_sample=True,max_new_tokens=MAX_NEW_TOKENS)
        generated_tokens = outputs[0][input_length:]
        
        # Print response
        print("\n ===================================================== \n")
        print(f"The Question: \n {question} \n")
        print(f"The Answer:")
        print(tokenizer.decode(generated_tokens.cpu().squeeze(),skip_special_tokens=True))
        print("\n ===================================================== \n")
            

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Define path to model')
    parser.add_argument('--model_path', metavar='model_path', 
                        help='the path to model')
    args = parser.parse_args()
    main(model_path=args.model_path)