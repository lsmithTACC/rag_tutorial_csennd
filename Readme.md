## Instructions:

We have configured the code such that it load an LLM backend from a local directory via the transformers library. This is probably the most transparent approach to loading an LLM backend, but keep in mind that more streamlined approaches exist (Ollama, vLLM, etc.). Before running the example, you will need to install the requirements:
```
pip install -r requirements.txt
```

And download the weights/config for the LLM backend and embedding models:
```
bash download_llm_backend.sh
```

Note that you can edit the .sh script with the LLM backend/install location of your choice.

Also, if you are running on an NVIDIA GPU, torch will sometimes download without CUDA support depending on your system config. You should ensure that the following python commands return True before running the script:

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