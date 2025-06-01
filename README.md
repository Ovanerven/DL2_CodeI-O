# DL2_CodeI-O: Code Reasoning with Logic-RL

This repository contains the DL2_CodeI-O codebase for training language models on code reasoning tasks using reinforcement learning. The project combines data generation from the **ReasonIO** module with the **Logic-RL** training framework to create models capable of reasoning about code input/output relationships.

**ReasonIO** is based on the CodeI/O approach from DeepSeek ([Li et al., 2025](https://arxiv.org/abs/2502.07316)), which systematically condenses diverse reasoning patterns embedded in code through input-output prediction tasks. Our implementation extends this approach with clustering-based diversity sampling and integration with reinforcement learning training.

## Repository Structure

```
DL2_CodeI-O/
├── Logic-RL-main/          # Modified Logic-RL framework for training
│   ├── reason_io_scripts/  # Training scripts for ReasonIO experiments
│   ├── verl/              # VERL reinforcement learning framework
│   │   └── utils/reward_score/
│   │       └── reason_io.py  # Custom reward function for ReasonIO tasks
│   ├── data/              # Training data directories
│   └── ...
├── ReasonIO/              # Data generation and preprocessing pipeline
│   ├── benchmark_evaluation/ # Evaluation scripts and benchmark datasets
│   ├── subsets/          # Generated dataset subsets
│   ├── clusters/         # Clustering results and embeddings
│   ├── subset_creation.ipynb  # Interactive clustering optimization notebook
│   ├── install_packages.py   # Package installation script
│   └── ...
└── plots/                # Analysis and visualization outputs
```

## Components Overview

### 1. ReasonIO (Data Generation)
The ReasonIO module handles the creation of diverse code reasoning datasets through:
- **Code preprocessing**: AST extraction and embedding generation via `preprocessing.py`
- **Interactive clustering**: Optimal clustering parameter exploration via `subset_creation.ipynb`
- **Clustering**: HDBSCAN-based clustering for diversity sampling
- **Input generator creation**: DeepSeek API-based generation of input generators for code problems
- **I/O pair generation**: Automatic input/output pair creation using generated input generators
- **Reasoning question generation**: Creation of deductive, abductive, and inductive reasoning tasks

### 2. Logic-RL-main (Training Framework)
Modified version of the original Logic-RL repository featuring:
- **Custom reward modeling**: Added `reason_io.py` reward function specifically for code reasoning tasks
- **REINFORCE++ training**: Policy optimization using REINFORCE++ advantage estimator
- **Multi-GPU training support** with FSDP (Fully Sharded Data Parallel)
- **Integration with VERL** framework for efficient RL training
- **Adapted training scripts** for ReasonIO dataset format

### 3. reason_io_scripts (Experiment Scripts)
Contains job scripts and shell scripts for running various training configurations:
- **REINFORCE++ training scripts** for different model sizes and epochs
- **Job submission files** for cluster/SLURM environments
- **Ablation study configurations**

## Key Modifications Made

### Logic-RL Framework Adaptations
The main modification to the original Logic-RL repository was creating a custom reward function:
- **`Logic-RL-main/verl/utils/reward_score/reason_io.py`**: Custom reward scoring function for code reasoning tasks
- **Training script adaptations**: Modified shell scripts in `reason_io_scripts/` to use ReasonIO dataset format
- **Data format compatibility**: Ensured training pipeline works with ReasonIO-generated Parquet files

All other components of Logic-RL remained unchanged.

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.1+ (for GPU training)
- 4x A100 80GB GPUs (recommended for full training)
- DeepSeek API key (for input generator creation)

### Setup Logic-RL Environment
```bash
cd Logic-RL-main
conda create -n logic python=3.9
conda activate logic

# Install PyTorch with CUDA support
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation

# Install VERL framework
pip install -e .

# Install additional dependencies
pip install wandb IPython matplotlib
pip install -r requirements.txt
```

### Setup ReasonIO Environment
```bash
cd ReasonIO
# Create environment from environment.yml file
conda env create -f environment.yml
conda activate reasonio

# Install packages required by code examples
python install_packages.py
```

## Usage

### Data Generation Pipeline

#### 1. Interactive Subset Creation (Recommended)
Use the Jupyter notebook for optimal clustering parameter exploration:
```bash
cd ReasonIO
conda activate reasonio
jupyter notebook subset_creation.ipynb
```
This notebook allows you to:
- Experiment with different clustering parameters
- Visualize clustering results
- Interactively select optimal subset size and diversity parameters
- Generate final subsets with optimal settings

#### 2. Generate Input Generators
Create input generator functions using DeepSeek API:
```bash
# Generate input generators for problems (REQUIRED BEFORE I/O pairs)
python generate_input_generators.py \
    --input_file subsets/ast-pyedur_full_subset_100_5_2000.jsonl \
    --output_dir input_generators \
    --api_key YOUR_DEEPSEEK_API_KEY \
    --temperature 0.0 \
    --max_workers 5
```

#### 3. Generate I/O Pairs
Create input/output pairs using the generated input generators:
```bash
# Generate I/O pairs from input generators (actual command used)
python generate_io_pairs.py \
    --input_file input_generators/input_generators.jsonl \
    --output_file input_generators/io_pairs_output.jsonl \
    --num_pairs 10 \
    --timeout 60
```

#### 4. Create Reasoning Questions
Generate different types of reasoning tasks in Logic-RL format:
```bash
# Create reasoning dataset with specific task types (actual command used)
python reasoning_questions.py \
    --input input_generators/io_pairs_output.jsonl \
    --logic_rl_format \
    --output ../Logic-RL-main/data/reason_io/deductive_abductive_only_ablation.parquet \
    --max_train_examples 2000 \
    --split_dataset \
    --train_ratio 0.8 \
    --shuffle_train \
    --task_types deductive abductive
```

### Model Training

#### Training with REINFORCE++
Run the main training script for ReasonIO:
```bash
cd Logic-RL-main
conda activate logic

# Run REINFORCE++ training with Qwen-2.5-7B model (5 epochs)
bash reason_io_scripts/reason_io_grpo_7b_5epoch.sh
```

#### Available Training Scripts
Located in `Logic-RL-main/reason_io_scripts/`:

- **`reason_io_grpo_7b_5epoch.sh`**: Main REINFORCE++ training (5 epochs, 7B model)
- **`reason_io_grpo_7b.sh`**: Extended REINFORCE++ training (7B model)
- **`reason_io_grpo_ablation.sh`**: Ablation study configuration
- **`kk_grpo_7b_5epoch.sh`**: Knights & Knaves dataset training

#### Job Submission Scripts
For cluster environments with SLURM:
```bash
# Submit training job
sbatch reason_io_scripts/reason_io_7b_5epoch.job
```

#### Training Configuration
Key training parameters (modifiable in shell scripts):
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Batch Size**: 8 (train/val)
- **Learning Rate**: 4e-7
- **Max Prompt Length**: 4096 tokens
- **Max Response Length**: 8192 tokens
- **GPUs**: 4x with tensor parallelism
- **Memory Optimization**: FSDP with offloading enabled

### Benchmark Evaluation

The repository includes comprehensive benchmarking scripts for evaluating trained models:

#### Available Benchmarks
Located in `ReasonIO/benchmark_evaluation/`:
- **`evaluate_benchmarks.py`**: Main evaluation script for various reasoning benchmarks
- **`test_reasonIO.py`**: ReasonIO-specific evaluation and testing
- **`amc.jsonl`**: AMC (American Mathematics Competitions) benchmark dataset
- **`aime_2021_2024.jsonl`**: AIME (American Invitational Mathematics Examination) dataset

#### Running Evaluations
```bash
cd ReasonIO/benchmark_evaluation
conda activate reasonio

# Evaluate trained models on benchmarks
python evaluate_benchmarks.py \
    --model_path path/to/trained/model \
    --benchmark amc \
    --output_file results.jsonl

# Test ReasonIO-specific tasks
python test_reasonIO.py \
    --model_path path/to/trained/model \
    --test_data your_test_data.jsonl
```

## Monitoring and Logging

### WandB Integration
Training progress is automatically logged to Weights & Biases:
- **Project Name**: `ReasonIO`
- **Experiment Naming**: Includes timestamp and configuration details
- **Metrics Tracked**: Loss, rewards, KL divergence, validation performance

### Log Files
Training logs are saved to:
```bash
Logic-RL-main/logs/reasonio/reason_io_${timestamp}.log
```

### Checkpoints
Model checkpoints are saved to:
```bash
Logic-RL-main/actor_checkpoints/reason_io_${timestamp}/
```

## Data Formats

### Input Data Format (JSONL)
```json
{
  "context": "Problem description with input/output requirements",
  "reference_code": "def solution():\n    # implementation",
  "input_generator": "def input_generator():\n    # generates inputs",
  "io_pairs": [
    {"input": {"x": 5}, "output": 25},
    {"input": {"x": 3}, "output": 9}
  ]
}
```

### Training Data Format (Parquet)
Logic-RL compatible format with:
- **prompt**: Formatted reasoning question
- **expected_output**: Target response
- **task_type**: deductive/abductive/inductive

## Performance and Requirements

### Hardware Requirements
- **Recommended**: 2x H100 80GB GPUs

### Dataset Sizes
- **Full dataset**: ~25GB (pyedur_full.jsonl)
- **Typical subset**: 2-7K examples (30-106MB)
- **Training data**: 2K examples (train) + 500 examples (val) for ablation studies

### API Requirements
- **DeepSeek API**: Required for input generator creation
- **Rate limiting**: Built-in retry logic with exponential backoff
- **Parallel processing**: Configurable worker count (default: 5)

## Citation

If you use this codebase, please cite the original Logic-RL paper and the CodeI/O paper that inspired our approach:

```bibtex
@misc{xie2025logicrlunleashingllmreasoning,
      title={Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning}, 
      author={Tian Xie and Zitian Gao and Qingnan Ren and Haoming Luo and Yuqian Hong and Bryan Dai and Joey Zhou and Kai Qiu and Zhirong Wu and Chong Luo},
      year={2025},
      eprint={2502.14768},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.14768}, 
}

@misc{li2025codeio,
      title={CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction}, 
      author={Junlong Li and Daya Guo and Dejian Yang and Runxin Xu and Yu Wu and Junxian He},
      year={2025},
      eprint={2502.07316},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.07316},
}
```

## Acknowledgments

- **Logic-RL**: Original framework by [Unakar/Logic-RL](https://github.com/Unakar/Logic-RL)
- **CodeI/O**: Foundational approach by [DeepSeek](https://arxiv.org/abs/2502.07316) for code input-output reasoning
- **VERL**: Reinforcement learning framework by [volcengine/verl](https://github.com/volcengine/verl)
- **CodeBERT**: Code embeddings by Microsoft Research
- **DeepSeek**: API for input generator creation

## License

This project combines code from multiple sources. Please refer to individual LICENSE files in subdirectories for specific licensing terms. 