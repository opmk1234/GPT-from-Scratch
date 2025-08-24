# GPT from Scratch (Character-Level Language Model)

This project is an educational implementation of a **GPT-like Transformer language model** from scratch using **PyTorch**.  
It follows the ideas from **Attention is All You Need (Vaswani et al., 2017)** and **GPT-1 (Radford et al., 2018)**, trained on *The Wizard of Oz* text.

## 🚀 Features
- Implemented key components manually:
  - Scaled Dot-Product Attention
  - Multi-Head Attention
  - Feed Forward Networks
  - Positional Embeddings
  - Transformer Blocks
- Character-level text encoding/decoding
- Training & evaluation loop
- Text generation with autoregressive decoding

## 📊 Results
After training for ~3000 iterations, the model learns to generate coherent character-level text in the style of *Wizard of Oz*.
