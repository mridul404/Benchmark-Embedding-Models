from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB

def benchmark_model(model_name, sentences):
    print(f"Benchmarking {model_name}")
    model = SentenceTransformer(model_name)
    
    # Model size and dimension
    model_size = get_model_size(model)
    dimension = model.get_sentence_embedding_dimension()
    
    # Embedding time
    start_time = time.time()
    embeddings = model.encode(sentences)
    embedding_time = (time.time() - start_time) / len(sentences)
    
    # Calculate average similarity
    similarities = cosine_similarity(embeddings)
    avg_similarity = np.mean(similarities)
    
    return {
        "Model Size (MB)": model_size,
        "Dimension": dimension,
        "Embedding Time (s/sentence)": embedding_time,
        "Average Similarity": avg_similarity
    }

# List of models to benchmark
models = [
    # Small dimension models
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/paraphrase-albert-small-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
    # Medium dimension models
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-roberta-large-v1",
    "sentence-transformers/stsb-roberta-base",
    # Large dimension models
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/stsb-roberta-large",
    "sentence-transformers/LaBSE",
    "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"
]

# Sample paragraphs for benchmarking
paragraphs = [
"The human brain is a marvel of biological engineering, containing approximately 86 billion neurons interconnected through trillions of synapses. This intricate network allows for complex cognitive functions, including memory, reasoning, and consciousness. Scientists continue to unravel the mysteries of the brain, exploring its plasticity and the ways in which it processes and stores information.",
               
"Climate change poses one of the greatest challenges to our planet's ecosystems and human societies. Rising global temperatures, driven by greenhouse gas emissions, lead to melting ice caps, rising sea levels, and increasingly extreme weather events. Addressing this crisis requires a multifaceted approach, including transitioning to renewable energy sources, implementing sustainable practices, and fostering international cooperation.",

"Artificial intelligence has made significant strides in recent years, with applications ranging from natural language processing to computer vision. Machine learning algorithms, particularly deep learning neural networks, have demonstrated remarkable capabilities in tasks such as image recognition, language translation, and even creative endeavors. As AI continues to advance, it raises important ethical questions about privacy, job displacement, and the future of human-machine interaction.",

"The field of quantum computing promises to revolutionize information processing by harnessing the principles of quantum mechanics. Unlike classical bits, quantum bits or qubits can exist in multiple states simultaneously, potentially allowing for exponentially faster computations for certain problems. While still in its early stages, quantum computing could have profound implications for cryptography, drug discovery, and complex system simulations.",

"The arts play a crucial role in human culture, serving as a medium for expression, reflection, and social commentary. From ancient cave paintings to contemporary digital art, artistic creations have evolved alongside human civilization. The arts not only provide aesthetic pleasure but also challenge our perceptions, provoke thought, and foster empathy by allowing us to experience diverse perspectives and emotions.",

"Space exploration continues to captivate human imagination and drive technological innovation. Recent developments, such as reusable rocket technology and plans for crewed missions to Mars, have reignited interest in space travel. Beyond the pursuit of scientific knowledge, space exploration offers potential solutions to terrestrial challenges, including resource scarcity and environmental monitoring.",

"The human microbiome, consisting of trillions of microorganisms living in and on our bodies, has emerged as a critical factor in health and disease. Research has revealed the microbiome's influence on various aspects of human physiology, including digestion, immunity, and even mental health. Understanding and manipulating the microbiome holds promise for developing new therapeutic approaches and personalized medicine strategies.",

"Blockchain technology, originally developed as the underlying system for cryptocurrencies, has found applications beyond digital finance. Its decentralized and transparent nature makes it suitable for various use cases, including supply chain management, voting systems, and digital identity verification. As blockchain continues to evolve, it has the potential to reshape industries and transform the way we conduct transactions and store information.",

"The field of genetics has undergone rapid advancement since the completion of the Human Genome Project. Technologies like CRISPR-Cas9 gene editing offer unprecedented precision in modifying DNA, opening up possibilities for treating genetic disorders and enhancing crop resilience. However, these developments also raise ethical concerns about the limits and implications of altering the genetic code of living organisms.",

"Urban planning and sustainable city development have become increasingly important as the world's population continues to concentrate in urban areas. Smart city initiatives leverage technology to improve efficiency, reduce environmental impact, and enhance quality of life for residents. Concepts such as green spaces, mixed-use developments, and intelligent transportation systems are being integrated to create more livable and resilient urban environments."
]

# Run benchmarks
results = {}
for model_name in models:
    results[model_name] = benchmark_model(model_name, paragraphs)

# Print results
print("\nBenchmark Results:")
print("-----------------")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")