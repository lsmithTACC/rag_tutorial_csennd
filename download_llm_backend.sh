# Activate env with all requirements downloaded
# source $SCRATCH/py-envs/rag/bin/activate

# Download model (Qwen3-4B is the default example, but it can be swapped)
#hf download Qwen/Qwen3-4B --local-dir ./models/Qwen3-4B
#hf download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
hf download Qwen/Qwen3-Embedding-0.6B --local-dir ./models/Qwen3-Embedding-0.6B
