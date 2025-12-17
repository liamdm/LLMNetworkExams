# LLMs can pass the CCNA Certification Exam But do they understand Networking?

This repository contains the code and experimental framework accompanying the paper:

> **LLMs can pass the CCNA Certification Exam But do they understand Networking?**

The goal of this project is to evaluate how well large language models (LLMs) can apply general-purpose knowledge to the specialized domain of computer networking, using professional certification exams as benchmarks.

---

## 📄 Paper Overview

Large language models (LLMs) have demonstrated strong performance across a wide range of tasks due to extensive pre-training on large-scale datasets. This raises an important question for applied and professional domains:

**Can LLMs meaningfully reason about specialized networking concepts at a level comparable to human experts?**

To investigate this, we evaluate multiple open-weight and closed-weight LLMs on questions drawn from three widely recognized industry certifications:

- **Cisco Certified Network Associate (CCNA)**
- **Cisco Certified Support Technician (CCST)**
- **Microsoft Networking Fundamentals (NF)**

These exams require not only factual recall, but also conceptual understanding and reasoning about networking protocols, architectures, and troubleshooting scenarios.

Our results show that while several models are capable of achieving passing scores—most notably GPT-4o and GPT-5.1 on CCNA—performance degrades on questions requiring higher-order reasoning. This suggests that current LLMs may still lack deep conceptual understanding in complex networking contexts.

---

## 🧠 What This Code Does

This codebase implements a **large-scale, reproducible evaluation pipeline** for assessing LLMs on networking exam questions.

At a high level, the pipeline performs:

1. **Question tagging and categorization**
2. **Difficulty rating**
3. **Automated question answering**
4. **Result storage and aggregation**
5. **Repeated trials across models, temperatures, and prompt configurations**

All experiments are designed to support statistical analysis and controlled comparisons across models.

---

## 📂 Project Structure

```
.  
├── main.py    # The code used in the paper for experiments
├── graphs.py  # The code used in the paper to do the analysis
└── README.md
```

---

## 🧪 Experimental Pipeline

### 1. Question Tagging & Categorization

For each exam dataset:

- Questions are tagged using a predefined taxonomy (<ticks>categories.txt<ticks>)
- Each question is assigned:
  - One or more **topic tags**
  - A **category (v2)**
  - A **difficulty rating**

Tagging and difficulty estimation are performed using a strong reference model to ensure consistency across datasets.

---

### 2. Question Answering Evaluation

Each question is answered under multiple controlled conditions:

- **Models**
  - GPT-5.1
  - GPT-4 / GPT-4 Turbo
  - GPT-3.5 variants
  - LLaMA 3 (8B via API)

- **Prompt configurations**
  - With and without explicit reasoning (“working”)
  - Different reasoning-effort levels (where supported)

- **Sampling parameters**
  - Multiple temperatures
  - Five independent repeats per configuration

This experimental design enables detailed analysis of **accuracy**, **variance**, and **robustness** across model families and prompting strategies.

---

## 💾 Output Artifacts

The pipeline produces several structured outputs:

- <ticks>rating_df.json<ticks>  
  Question-level tags, categories, and difficulty ratings across exams and models.

- <ticks>*.ndjson<ticks>  
  Incrementally saved model responses using a normalized row-key format, optimized for large-scale runs.

- <ticks>qa_df_all.csv<ticks> 
  Old format of answer storage - we recommend not using this, although for compatibility with graphs.csv we leave this intact. Stores all models together in one big CSV.

---

## 🔁 Reproducibility Notes

- All experiments are deterministic **per repeat** via request caching.
- Threaded execution is used to accelerate large runs while preserving correctness.
- The framework supports resuming interrupted runs without recomputing completed samples.

---

## 📊 Intended Use

This repository is intended for:

- Researchers studying LLM reasoning in specialized technical domains
- Comparative evaluation of open-weight vs closed-weight models
- Analysis of prompting, temperature, and reasoning controls
- Reproducible benchmarking against professional certification standards

---

## 📜 License & Disclaimer

This project is provided for **research and evaluation purposes only**.  

Due to licensing and intellectual property restrictions, **we do not publish or redistribute the certification question datasets**, as they are commercially owned by their respective vendors. However, researchers and practitioners who legally obtain these materials may reproduce our experiments by placing their purchased datasets into the expected directory structure and running the evaluation pipeline themselves.
