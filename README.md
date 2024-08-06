# Training a Transformer Model for Text Generation

This document outlines the process for training a transformer model for text generation using PyTorch. The model learns patterns from a dataset (e.g., tiny Shakespeare) and generates new text based on these patterns. The approach is creaed from Andrej Karpathy's YouTube video on nano-GPT.

## Requirements

- Python 3.x
- PyTorch
- CUDA (Optional, for GPU acceleration)

## Dataset

The model is trained on a dataset that can be obtained with the following command:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt