This is a Stanford XCS224N project where I was tasked to build a GPT-style
Transformer from scratch to explore pretraining and transfer learning.

Core Task

Given a person's name, predict their birthplace — a knowledge-intensive task
that benefits from pretraining on Wikipedia text.

Key Components

┌──────────────────┬─────────────────────────────────────────────────┐
│       File       │                     Purpose                     │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/model.py     │ GPT architecture (embeddings, transformer       │
│                  │ blocks, output head)                            │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/attention.py │ Causal self-attention and cross-attention       │
│                  │ mechanisms                                      │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/dataset.py   │ Span corruption pretraining dataset +           │
│                  │ name→birthplace finetuning dataset              │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/trainer.py   │ Training loop with LR scheduling,               │
│                  │ checkpointing                                   │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/helper.py    │ Model initialization, pretrain/finetune         │
│                  │ orchestration                                   │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/utils.py     │ Seed setting, autoregressive sampling,          │
│                  │ birthplace prediction evaluation                │
├──────────────────┼─────────────────────────────────────────────────┤
│ src/run.py       │ CLI entry point                                 │
└──────────────────┴─────────────────────────────────────────────────┘

Implemention plan:

1. Span corruption — a pretraining objective that randomly masks spans in
Wikipedia text and trains the model to reconstruct them
2. Fine-tuning pipeline — train on name→birthplace pairs, with and without
pretraining
3. Perceiver architecture — an efficient variant using cross-attention to
compress sequence length (reduces O(n²) to O(n·m))

ML Pipeline

Pretrain (Wikipedia, 650 epochs) → Fine-tune (birthplace data, 10 epochs) →
Evaluate accuracy on test set

Model Specs

- 4 transformer layers, 8 attention heads, 256-dim embeddings
- Character-level tokenization, context window of 128
- Based on Karpathy's minGPT

Dependencies

PyTorch, NumPy, tqdm, TensorBoard, and other standard ML libraries (Python
3.8, conda environment).

## Running the Project

### 1. Set up the environment

```bash
cd src

# CPU only
conda env create -f environment.yml
conda activate Transformer

# OR with GPU (CUDA 11.8)
conda env create -f environment_cuda.yml
conda activate Transformer
```

### 2. Run the pipeline

The easiest way is using the provided `run.sh` script from the `src/` directory:

**Quick start — finetune without pretraining (~minutes):**

```bash
./run.sh vanilla_finetune_without_pretrain
./run.sh vanilla_eval_dev_without_pretrain
```

**Full pipeline — pretrain on Wikipedia then finetune (~2 hours):**

```bash
./run.sh vanilla_pretrain
./run.sh vanilla_finetune_with_pretrain
./run.sh vanilla_eval_dev_with_pretrain
./run.sh vanilla_eval_test_with_pretrain
```

**Perceiver variant (optional):**

```bash
./run.sh perceiver_pretrain
./run.sh perceiver_finetune_with_pretrain
./run.sh perceiver_eval_dev_with_pretrain
```

### 3. Or use the Python CLI directly

```bash
# Finetune without pretraining
python run.py --function=finetune --variant=vanilla \
  --pretrain_corpus_path=./data/wiki.txt \
  --writing_params_path=./vanilla.model.params \
  --finetune_corpus_path=./data/birth_places_train.tsv

# Evaluate on dev set
python run.py --function=evaluate --variant=vanilla \
  --pretrain_corpus_path=./data/wiki.txt \
  --reading_params_path=./vanilla.model.params \
  --eval_corpus_path=./data/birth_dev.tsv \
  --outputs_path=./vanilla.nopretrain.dev.predictions
```

### 4. View training logs

```bash
tensorboard --logdir=expt/
```
