# Activate env with all requirements downloaded
# Replace the path here with the path to your virtual env
# source $SCRATCH/py-envs/rag/bin/activate

# Download LLM Backend (Qwen3-8B is a good starting point)
# Note: the way you call the huggingface API may change depending on your 
# system config and python version. Newer versions use the 'hf' command, while 
# others use the 'huggingface-cli' command.
#hf download Qwen/Qwen3-4B --local-dir ./models/Qwen3-4B
#hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B

# Download Embedding Model
#hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/Qwen3-Embedding-0.6B
