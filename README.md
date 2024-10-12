# Embedding Models Benchmark

I conducted this benchmarking for my personal reference when implementing Retrieval-Augmented Generation (RAG) using free embedding models from HuggingFace. This repository contains benchmarks for various embedding models from the Hugging Face sentence-transformers library. The benchmarks compare model size, embedding dimension, embedding time, and average similarity.

I mainly choose smaller size models for using locally.

NOTE: A Comprehensive Benchmark already exist in [📊 HuggingFace](https://huggingface.co/spaces/mteb/leaderboard).

## Benchmark Results

| Model | Size (MB) | Dimension | Embedding Time (s/paragraph) | Avg Similarity |
|-------|-----------|-----------|-----------------------------| ---------------|
| paraphrase-albert-small-v2 | 44.57 | 768 | 0.030840 | 0.459125 |
| paraphrase-MiniLM-L3-v2 | 66.34 | 384 | 0.007843 | 0.442153 |
| all-MiniLM-L6-v2 | 86.64 | 384 | 0.011985 | 0.521834 |
| all-distilroberta-v1 | 313.26 | 768 | 0.023317 | 0.557498 |
| all-mpnet-base-v2 | 417.66 | 768 | 0.068026 | 0.575438 |
| multi-qa-mpnet-base-dot-v1 | 417.66 | 768 | 0.045019 | 0.656036 |
| paraphrase-multilingual-MiniLM-L12-v2 | 448.81 | 384 | 0.021847 | 0.491451 |
| stsb-roberta-base | 475.49 | 768 | 0.054983 | 0.441052 |
| distilbert-multilingual-nli-stsb-quora-ranking | 513.97 | 768 | 0.024070 | 0.822345 |
| distiluse-base-multilingual-cased-v2 | 515.47 | 512 | 0.020928 | 0.554773 |
| paraphrase-multilingual-mpnet-base-v2 | 1060.65 | 768 | 0.074159 | 0.571580 |
| all-roberta-large-v1 | 1355.59 | 1024 | 0.216250 | 0.591911 |
| stsb-roberta-large | 1355.59 | 1024 | 0.205648 | 0.489088 |
| LaBSE | 1798.70 | 768 | 0.042616 | 0.567270 |

## Interpretation

- **Model Size**: Ranges from 44.57 MB (paraphrase-albert-small-v2) to 1798.70 MB (LaBSE).
- **Dimension**: Most models use 384 or 768 dimensions, with a few using 512 or 1024.
- **Embedding Time**: Varies from 0.007843 s/sentence (paraphrase-MiniLM-L3-v2) to 0.216250 s/sentence (all-roberta-large-v1).
- **Average Similarity**: Ranges from 0.441052 (stsb-roberta-base) to 0.822345 (distilbert-multilingual-nli-stsb-quora-ranking).

## Observations

1. Smaller models like paraphrase-albert-small-v2 and paraphrase-MiniLM-L3-v2 offer the advantage of lower storage requirements and generally faster embedding times.
2. The paraphrase-MiniLM-L3-v2 model stands out among smaller models with the fastest embedding time of 0.007843 s/sentence.
3. Larger models like all-roberta-large-v1 and LaBSE have higher dimensions and model sizes, which generally results in slower embedding times but may offer better performance in some tasks.
4. The distilbert-multilingual-nli-stsb-quora-ranking model has the highest average similarity score (0.822345) while maintaining a moderate size and embedding time, making it an interesting option to consider.
5. There's often a trade-off between model size/dimension and embedding time, but this doesn't always correlate directly with the average similarity score.

## Conclusion

The choice of embedding model depends on the specific requirements of your task, considering factors such as computational resources, speed requirements, and the importance of embedding quality (as reflected in the average similarity score). This benchmark provides a starting point for selecting an appropriate model based on these criteria, particularly for RAG implementations using free models from HuggingFace.

For local use, smaller models like paraphrase-albert-small-v2, paraphrase-MiniLM-L3-v2, and all-MiniLM-L6-v2 offer a good balance between size, speed, and performance. However, if storage and computational resources allow, models like distilbert-multilingual-nli-stsb-quora-ranking might provide better embedding quality for certain applications.
