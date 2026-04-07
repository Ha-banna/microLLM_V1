# Character-Level Bigram Language Model

This project implements a foundational **Bigram Language Model** trained on the text of *"The Wizard of Oz."* It serves as an entry-level exploration into how neural networks process language by converting characters into numerical vectors and predicting the next character in a sequence.

## 🚀 Overview
The model follows a classic NLP pipeline:
1.  **Tokenization**: Converts raw strings into character-level tokens.
2.  **Embedding**: Maps tokens to vectors using an embedding table.
3.  **Next-Token Prediction**: Uses a simple neural network to predict the probability of the next character based solely on the current one.
4.  **Generation**: Produces new text by repeatedly sampling from the model's predicted probability distributions.

## 🛠️ Technical Specifications
* **Architecture**: `BigramModel` class inheriting from `torch.nn.Module`.
* **Framework**: PyTorch.
* **Training Hardware**: Supports both CPU and NVIDIA GPU (CUDA).
* **Hyperparameters**:
    * `block_size`: 100 (context window).
    * `batch_size`: 1000.
    * `learning_rate`: 1e-2.
    * `training_loops`: 1000.

## 📂 Project Structure
* **Data Processing**: The script reads `wizard of oz.txt`, creates a vocabulary of unique characters, and builds mapping dictionaries (`char_to_int` and `int_to_char`).
* **Model**: A simple lookup table (`nn.Embedding`) where the "logits" are effectively the scores for what character comes next.
* **Optimization**: Uses the `AdamW` optimizer and `cross_entropy` loss function to refine predictions.

## 📈 Results
After 1,000 training loops, the model's loss decreased to approximately **2.41**. While a Bigram model only looks at one character of context—resulting in "gibberish" that resembles the source language's structure—it successfully learns basic word patterns and spacing.

### Example Output:
> *"pleacenois this fan u'varimareot gheeowowingateareg om't r orewertode Jitha owerit..."*

## 🏗️ Future Improvements for Later Models
This project serves as a stepping stone toward more advanced architectures:
* **Self-Attention**: Implementing a Transformer block to allow the model to look further back in the sequence.
* **Subword Tokenization**: Moving from characters to Byte Pair Encoding (BPE) for better efficiency.
* **Scaling**: Training on larger datasets with Multi-Head Attention.
