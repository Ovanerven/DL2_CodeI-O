#!/usr/bin/env python
"""
Code-IO Dataset Preprocessing Script

This script preprocesses a large code dataset by:
1. Extracting code from prompts and converting to AST
2. Generating code embeddings using CodeBERT
3. Reducing dimensionality with PCA
4. Using pre-computed HDBSCAN clusters
5. Sampling 10K diverse examples across clusters
6. Creating a subset JSONL file

Usage:
    python preprocessing.py --input pyedur_full.jsonl --output final_subset.jsonl [--target_size 10000]
"""

import argparse
import json
import numpy as np
import os
import ast
import torch
import hdbscan
from joblib import Memory
from tqdm import tqdm
from time import time
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Preprocess code dataset into a diverse subset")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output subset JSONL file")
    parser.add_argument("--ast_path", type=str, help="Path to AST-processed JSONL file (will create if not exists)")
    parser.add_argument("--embedding_path", type=str, default="clusters/all_embeddings.npy", help="Path to save/load embeddings")
    parser.add_argument("--reduced_embedding_path", type=str, default="clusters/reduced_embeddings.npy", help="Path to save/load reduced embeddings")
    parser.add_argument("--labels_path", type=str, default="clusters/hdbscan_labels_reduced10_mcs20_ms5.npy", help="Path to HDBSCAN clustering labels")
    parser.add_argument("--probabilities_path", type=str, default="clusters/hdbscan_probabilities_reduced10_mcs20_ms5.npy", help="Path to HDBSCAN probabilities")
    parser.add_argument("--target_size", type=int, default=10000, help="Target number of samples in subset")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip AST processing step")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding generation step")
    parser.add_argument("--skip_clustering", action="store_true", help="Skip clustering step (use existing labels)")
    parser.add_argument("--dimension", type=int, default=10, help="Number of dimensions for PCA reduction")
    parser.add_argument("--min_cluster_size", type=int, default=100, help="Min cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5, help="Min samples for HDBSCAN")
    
    return parser.parse_args()


def count_lines(filepath):
    """Count the number of lines in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def extract_code(prompt):
    """Extract code from a prompt"""
    marker = "You can refer to this code to guide your reasoning but not copy spans of code directly."
    if marker in prompt:
        return prompt.split(marker, 1)[-1].strip()
    return ""


def code_to_ast(code):
    """Convert code to AST representation"""
    try:
        tree = ast.parse(code)
        return ast.dump(tree)
    except SyntaxError:
        return None


def process_jsonl(input_path, output_path, buffer_size=5000):
    """Process JSONL file to extract code and generate AST"""
    if os.path.exists(output_path):
        print(f"AST-processed file already exists at {output_path}")
        return output_path
        
    buffer = []
    total_lines = count_lines(input_path)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for idx, line in enumerate(tqdm(infile, total=total_lines, desc="Extracting code & generating AST")):
            try:
                data = json.loads(line)
                prompt = data.get("prompt", "")
                code = extract_code(prompt)
                data["code"] = code

                # Convert to AST
                ast_representation = code_to_ast(code)
                if ast_representation is not None:
                    data["ast"] = ast_representation

                buffer.append(json.dumps(data))

                # Write in chunks
                if len(buffer) >= buffer_size:
                    outfile.write("\n".join(buffer) + "\n")
                    buffer.clear()

            except Exception as e:
                continue

        # Write any remaining lines
        if buffer:
            outfile.write("\n".join(buffer) + "\n")
    
    print(f"AST processing complete: {output_path}")
    return output_path


def generate_code_embeddings(jsonl_path, embedding_path='embeddings.npy', batch_size=64, max_length=512):
    """Generate code embeddings using CodeBERT"""
    if os.path.exists(embedding_path):
        print(f"Loading existing embeddings from {embedding_path}")
        return np.load(embedding_path)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base").cuda()
    embeddings = []
    total = count_lines(jsonl_path)
    chunked_count = 0

    def process_code_chunk(batch):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to('cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        return torch.mean(outputs.last_hidden_state, dim=1).cpu()

    with open(jsonl_path) as f:
        for line in tqdm(f, desc="Generating embeddings", total=total):
            entry = json.loads(line)
            code = entry['code']
            tokens = tokenizer.tokenize(code)

            if len(tokens) <= max_length:
                # Code fits within limit
                chunk_embeddings = process_code_chunk([code])
                embeddings.append(chunk_embeddings[0].numpy())
            else:
                # Code exceeds limit - use chunking
                chunked_count += 1
                stride = max_length // 2
                chunk_inputs = []

                for i in range(0, len(tokens), stride):
                    chunk_tokens = tokens[i:i+max_length]
                    if len(chunk_tokens) < max_length // 4:  # Skip small final chunks
                        continue
                    chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
                    chunk_inputs.append(chunk_text)

                if not chunk_inputs:  # Ensure we have at least one chunk
                    chunk_inputs = [tokenizer.convert_tokens_to_string(tokens[:max_length])]

                # Process each chunk in batches
                all_chunk_embeddings = []
                for i in range(0, len(chunk_inputs), batch_size):
                    batch_chunks = chunk_inputs[i:i+batch_size]
                    batch_embs = process_code_chunk(batch_chunks)
                    all_chunk_embeddings.extend([emb for emb in batch_embs])

                # Aggregate the chunk embeddings
                all_chunk_embeddings = torch.stack(all_chunk_embeddings)
                final_embedding = torch.mean(all_chunk_embeddings, dim=0).numpy()
                embeddings.append(final_embedding)

    print(f"Embedding generation complete ({chunked_count} examples required chunking)")
    embeddings = np.array(embeddings)
    np.save(embedding_path, embeddings)
    return embeddings


def reduce_dimensions(embeddings, n_components=10, output_path=None):
    """Reduce embedding dimensions using PCA"""
    if output_path and os.path.exists(output_path):
        print(f"Loading existing reduced embeddings from {output_path}")
        return np.load(output_path)
        
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Report explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    print(f"Cumulative explained variance with {n_components} components: {cumulative_variance[-1]:.2%}")
    
    if output_path:
        np.save(output_path, reduced_embeddings)
        print(f"Saved reduced embeddings to {output_path}")
        
    return reduced_embeddings


def run_hdbscan_clustering(data, min_cluster_size=20, min_samples=5, labels_path=None, probs_path=None):
    """Run HDBSCAN clustering or load existing results"""
    # Check if we can load existing results
    if labels_path and os.path.exists(labels_path) and probs_path and os.path.exists(probs_path):
        print(f"Loading existing clustering results from {labels_path} and {probs_path}")
        labels = np.load(labels_path)
        probabilities = np.load(probs_path)
        return labels, probabilities
    
    # Set up caching
    memory = Memory(location='./hdbscan_cache', verbose=0)
    
    @memory.cache
    def cached_hdbscan_fit(data, min_cluster_size, min_samples):
        print(f"Running HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        start_time = time()
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            core_dist_n_jobs=-1,
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(data)
        print(f"Clustering completed in {time() - start_time:.2f} seconds")
        return clusterer, labels
    
    # Run clustering
    clusterer, labels = cached_hdbscan_fit(data, min_cluster_size, min_samples)
    probabilities = clusterer.probabilities_
    
    # Save results if paths provided
    if labels_path:
        np.save(labels_path, labels)
    if probs_path:
        np.save(probs_path, probabilities)
    
    # Print basic statistics
    n_clusters = len(np.unique(labels)) - 1  # -1 for noise
    noise_points = np.sum(labels == -1)
    noise_ratio = noise_points / len(labels)
    print(f"Clustering results: {n_clusters} clusters, {noise_ratio:.2%} noise points")
    
    return labels, probabilities


def diversity_sampling(embeddings, labels, target_samples=10000, query_ratio=0.1, min_cluster_samples=1):
    """Sample diverse points from clusters to create a subset"""
    # Get non-noise clusters and their sizes
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters != -1]
    cluster_sizes = {cid: np.sum(labels == cid) for cid in unique_clusters}
    total_non_noise = sum(cluster_sizes.values())
    
    # Calculate effective compression ratio
    effective_ratio = target_samples / total_non_noise
    print(f"Sampling {target_samples} points from {total_non_noise} non-noise points")
    
    # Calculate quota for each cluster
    quotas = {cid: max(min_cluster_samples, int(effective_ratio * size)) 
              for cid, size in cluster_sizes.items()}
    
    # Adjust quotas to match target_samples
    quota_sum = sum(quotas.values())
    remaining = target_samples - quota_sum
    
    if remaining != 0:
        # Distribute remaining samples proportionally to cluster size
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        for i in range(abs(remaining)):
            cid = sorted_clusters[i % len(sorted_clusters)][0]
            quotas[cid] += 1 if remaining > 0 else -1
            if quotas[cid] <= 0:
                quotas[cid] = 1
    
    # Process clusters
    selected_indices = []
    success_count = 0
    fallback_count = 0
    
    for cluster_id in tqdm(unique_clusters, desc="Sampling diverse points"):
        quota = quotas[cluster_id]
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_mask]
        n_points = len(cluster_embeddings)
        
        if n_points <= quota:
            # Take all points if quota >= cluster size
            selected_indices.extend(cluster_indices)
            continue
            
        # Select query set
        query_size = max(1, min(int(query_ratio * n_points), n_points // 2))
        query_indices = np.random.choice(n_points, size=query_size, replace=False)
        K = cluster_embeddings[query_indices]
        
        try:
            # Compute diversity scores
            similarity_matrix = np.dot(cluster_embeddings, K.T)
            distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
            diversity_scores = np.min(distance_matrix, axis=1)
            
            # Handle points with zero diversity (typically query points themselves)
            if np.any(diversity_scores == 0):
                mean_diversity = np.mean(diversity_scores[diversity_scores > 0]) if np.any(diversity_scores > 0) else 0.1
                diversity_scores[diversity_scores == 0] = mean_diversity / 2
            
            # Sample points proportionally to diversity
            probabilities = diversity_scores / np.sum(diversity_scores)
            probabilities = probabilities / np.sum(probabilities)  # Ensure exact sum to 1
            
            # Determine whether to use probability-based sampling or just take top-K
            if n_points > 5 * quota:  # For large clusters, probability sampling
                selected = np.random.choice(
                    n_points, 
                    size=quota, 
                    replace=False, 
                    p=probabilities
                )
            else:
                # For smaller clusters, just take the top-K most diverse points
                selected = np.argsort(-diversity_scores)[:quota]
                
            selected_indices.extend(cluster_indices[selected])
            success_count += 1
            
        except Exception:
            # Fallback to random sampling on error
            selected = np.random.choice(n_points, size=quota, replace=False)
            selected_indices.extend(cluster_indices[selected])
            fallback_count += 1

    # Final validation
    selected_indices = np.array(selected_indices)
    
    # Trim if needed to match exact target
    if len(selected_indices) > target_samples:
        excess = len(selected_indices) - target_samples
        to_remove = np.random.choice(len(selected_indices), size=excess, replace=False)
        mask = np.ones(len(selected_indices), dtype=bool)
        mask[to_remove] = False
        selected_indices = selected_indices[mask]
    
    print(f"Selected {len(selected_indices)} samples using diversity sampling")
    print(f"Success: {success_count} clusters, Fallback: {fallback_count} clusters")
    
    return selected_indices


def create_subset_jsonl(original_jsonl_path, sampled_indices, output_jsonl_path):
    """Create a subset JSONL file using the sampled indices"""
    # Sort indices for sequential reading
    sorted_indices = np.sort(sampled_indices)
    
    # Process file in a single pass
    current_idx = 0  # Track position in the sorted_indices array
    current_line = 0  # Track position in the file
    
    with open(original_jsonl_path, 'r', encoding='utf-8') as infile, open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        pbar = tqdm(total=len(sorted_indices), desc="Creating subset file")
        
        for line in infile:
            # Check if we need this line
            if current_idx < len(sorted_indices) and current_line == sorted_indices[current_idx]:
                outfile.write(line)
                current_idx += 1
                pbar.update(1)
                
                # Early termination if we've found all indices
                if current_idx >= len(sorted_indices):
                    break
            
            # Move to next line
            current_line += 1
        
        pbar.close()
    
    print(f"Created subset with {current_idx} samples at {output_jsonl_path}")
    return output_jsonl_path


def main():
    """Main execution flow"""
    args = parse_args()
    
    # Step 1: Process JSONL to extract code and generate AST
    if not args.skip_preprocessing:
        ast_path = args.ast_path or f"ast-{os.path.basename(args.input)}"
        jsonl_path = process_jsonl(args.input, ast_path)
    else:
        jsonl_path = args.ast_path or args.input
        print(f"Skipping preprocessing, using {jsonl_path}")
    
    # Step 2: Generate code embeddings
    if not args.skip_embeddings:
        embeddings = generate_code_embeddings(jsonl_path, args.embedding_path)
        normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
    else:
        print(f"Loading embeddings from {args.embedding_path}")
        embeddings = np.load(args.embedding_path)
        normalized_embeddings = normalize(embeddings, norm='l2', axis=1)
    
    # Step 3: Reduce dimensionality
    reduced_embeddings = reduce_dimensions(
        normalized_embeddings, 
        n_components=args.dimension,
        output_path=args.reduced_embedding_path
    )
    
    # Step 4: Get cluster labels
    if not args.skip_clustering:
        labels, probabilities = run_hdbscan_clustering(
            reduced_embeddings,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            labels_path=args.labels_path,
            probs_path=args.probabilities_path
        )
    else:
        print(f"Loading existing clustering labels from {args.labels_path}")
        labels = np.load(args.labels_path)
        probabilities = np.load(args.probabilities_path)
    
    # Step 5: Diversity sampling
    sampled_indices = diversity_sampling(
        reduced_embeddings,
        labels,
        target_samples=args.target_size
    )
    
    # Step 6: Create subset JSONL
    create_subset_jsonl(jsonl_path, sampled_indices, args.output)
    
    print(f"Pipeline complete. Subset saved to: {args.output}")


if __name__ == "__main__":
    main()