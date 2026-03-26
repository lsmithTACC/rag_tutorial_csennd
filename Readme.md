## Instructions:

This codebase provides a simple implementation of retrieval augmented generation (RAG). We have set up the code to search a long academic paper ("A foundation model for the earth system" by Bodnar et al., which is a great read if you have time), and return relevant portions of the paper back into the model's context for use when responding to a prompt.

We have configured the code such that it load an LLM backend from a local directory via the transformers library. This is probably the most transparent approach to loading an LLM backend, but keep in mind that more streamlined approaches do exist (Ollama, vLLM, etc.). Before running the example, you will need to install the requirements:
```
pip install -r requirements.txt
```

And download the weights/config for the LLM backend and embedding models:
```
bash download_llm_backend.sh
```

Note that you can edit the .sh script with the LLM backend/install location of your choice.

If you are running on an NVIDIA GPU, torch will sometimes download without CUDA support depending on your system config. You should ensure that the following python commands return True before running the script:

```
import torch
torch.cuda.is_available()
```

Once requirements are downloaded, run the example via:

```
python rag_example.py --model_path=<your_path>
```

Where <your_path> is replaced by the path to your local model directory, as specified in the .sh script.

The example should return something like the following:

```text
 ===================================================== 

The Question: 
 Desribe Aurora's neural network architecture. 

The Answer:
 Aurora's neural network architecture consists of an encoder and decoder module, with a 3D latent representation. The encoder tokenizes and compresses input weather states into a 3D latent space using Perceiver-style cross-attention blocks, while the decoder reconstructs target output variables in spatial patches by decoding the 3D...
```

In addition to the LLM backend specified by --model_path, the code features a number of user controls that will need to be tuned to your specific document corpus. These controls are noted at the top of rag_example.py and described in more detail below. We've roughly organized them from most to least important.

EMBEDDING_MODEL - determines the embedding model used to convert document segments into database vectors. There are a few common options listed in download_llm_backend, and many more are available via HuggingFace. Note that both your embedding model and your LLM backend will need to fit within your GPU's memory. Our default embedding model (Qwen3-Embedding-0.6B) is a reasonable choice for a single GPU with ~40 GB of memory.

CHUNK_SIZE - determines the number of characters allocated to each segment during text splitting. We find that values on the order of 1e3 tend to be appropriate for most technical documents. Note that this parameter is given in characters, not tokens (hence its large numerical value).

CHUNK_OVERLAP - determines the overlap between adjacent segments during text splitting. The typical recommendation is that overlap should be about 10% of your chunk_size. Again, this parameter is given in characters, not tokens.

NUM_K - determines the number of segments returned from the vector store to the model's context. This value generally varies between 1 and 10. We find that lower numbers (2-4) are usually sufficient, but you may want to experiment with higher values if your LLM's answers appear too specific. Just note that a higher NUM_K leads to more computational overhead.

MAX_NEW_TOKENS - determines the maximum number of tokens with which the LLM can respond to the user's prompt. Lower this value if your LLM responses are too verbose, raise it if the LLM responses are too short. The code is configured to simply 'cut-off' the LLM once it exceeds MAX_NEW_TOKENS.

