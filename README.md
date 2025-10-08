Contributors:
Mohamad Idriss,
Sriram Acharya Mudumbai,
Vibin Chandrabose,
Walid El Mahdy,
Rohit Jacob Issac

This comprehensive script is a complete end-to-end pipeline for fine-tuning a small language model (SLM), building a knowledge base for it to use, interacting with the model via a RAG (Retrieval-Augmented Generation) system, and finally, evaluating its performance with multiple benchmarks.

Here’s a breakdown of each major section:

### **1. Setup and Initialization**

  * **Install Libraries:** It begins by installing all the necessary Python packages. This includes `transformers` for the models, `datasets` for data, `bitsandbytes` for efficient model loading, and specialized libraries like `faiss-cpu` for vector search and `evaluate` for performance metrics.
  * **Mount Google Drive:** It connects to your Google Drive. This is a crucial step for a Colab notebook, as it allows the script to save large files (like the final model and the vector database) persistently.
  * **Import Libraries:** It imports all the required modules from the installed packages, making them ready for use.

### **2. Model Fine-Tuning**

This section takes a pre-trained small language model and further trains it on a specific task.

  * **Configuration:** It defines a `ProjectConfig` class to hold all important settings in one place, such as file paths, model names (`distilgpt2`), the dataset to use (`squad`), and training parameters like learning rate and batch size.
  * **Load Base Model:** It downloads `distilgpt2`, a smaller, distilled version of GPT-2, and its tokenizer. This model is chosen because it's small enough to train quickly.
  * **Prepare SQuAD Dataset:** It loads the Stanford Question Answering Dataset (SQuAD). A custom function then formats this data into a specific "RAG-style" prompt, which looks like this:
    ```
    Context: [some paragraph]
    Instruction: [a question about the paragraph]
    Response: [the answer]
    ```
    Training the model on this format teaches it to answer questions based on a provided context, which is the core idea behind RAG.
  * **Training:** It uses the Hugging Face `Trainer` to fine-tune `distilgpt2` on the formatted SQuAD dataset. The script is smart enough to resume from a previous checkpoint if the training process was interrupted.
  * **Save Model:** Once training is complete, the newly fine-tuned model is saved to your Google Drive.

### **3. Building the Vector Database**

This is where the "Retrieval" part of RAG is built. The goal is to create a searchable knowledge base.

  * **Load Wikipedia Data:** It streams the official Wikipedia dataset. Streaming is used to avoid downloading the entire massive dataset, saving memory.
  * **Chunking:** The text from Wikipedia articles is broken down into smaller, manageable chunks. This is important because language models have a limited context window.
  * **Create Embeddings:** It loads a `sentence-transformer` model (`all-MiniLM-L6-v2`), which is excellent at converting text into numerical vectors (embeddings). It then processes all the text chunks to create a vector for each one.
  * **Build FAISS Index:** It uses FAISS, a library from Facebook AI for efficient similarity search. It creates an index from all the text embeddings. This index is highly optimized and allows for incredibly fast searching over millions of vectors.
  * **Save to Drive:** Both the FAISS index and the original text chunks are saved to your Google Drive.

### **4. Interactive RAG Chat**

This section brings everything together into a functional chatbot.

  * **Load Components:** It loads the fine-tuned model, the embedding model, the FAISS index, and the knowledge base from your Google Drive.
  * **Semantic Search Function (`get_semantic_rag_response`):** This is the core of the RAG system. When you ask a question:
    1.  It converts your question into a vector embedding.
    2.  It uses the FAISS index to instantly find the `k` most similar text chunks from the Wikipedia knowledge base.
    3.  It constructs a new prompt using the template from the training phase, injecting the retrieved text chunks as the "Context."
    4.  It sends this complete prompt (context + question) to your fine-tuned model.
    5.  The model, having been trained on this format, generates an answer based on the provided context.
  * **Chat Loop:** It starts an interactive loop where you can ask questions and get answers from the RAG-powered model.

### **5. Model Evaluation**

This is the scientific part of the script, where the model's performance is rigorously measured.

  * **Custom Metrics:**
      * It loads the SQuAD validation set (data the model has never seen).
      * It generates answers for a sample of these questions.
      * It calculates several text quality scores:
          * **ROUGE:** Measures overlap between the generated answer and the true answer.
          * **BLEU:** Commonly used for translation quality, it measures precision.
          * **BERTScore:** A more advanced metric that uses embeddings to check if the generated and true answers are semantically similar.
      * It also measures efficiency by calculating the average inference time per answer.
  * **Standardized Benchmarks:**
      * It installs the **LM Evaluation Harness**, a standard tool for benchmarking language models.
      * It runs your model against a set of well-known academic benchmarks:
          * **HellaSwag:** Commonsense reasoning.
          * **TruthfulQA:** Measures a model's tendency to generate truthful answers vs. common misconceptions.
          * **MMLU:** A massive multitask test covering 57 different subjects.
          * **HumanEval:** A test for code generation.
      * Finally, it displays a summary of all the benchmark results, giving you a clear picture of your model's capabilities across different domains.
   
  * **QLORA with RAG Results:**
  * QLORA Model Performance
    
    <img width="509" height="82" alt="Screenshot 2025-10-08 at 6 34 13 PM" src="https://github.com/user-attachments/assets/e0cbd85d-d242-4018-b3d9-ceca1106eccc" />

    QlORA Model Training Results
    <img width="1138" height="523" alt="Screenshot 2025-10-08 at 6 32 51 PM" src="https://github.com/user-attachments/assets/fc22f39e-195a-47dc-b4b0-87076fd52bd3" />

    Testing Results - NO RAG

    <img width="1642" height="402" alt="image" src="https://github.com/user-attachments/assets/aa113a75-9aa5-456d-84fb-1e2ca2fd9be2" />

    RAG testing Results
    <img width="1608" height="555" alt="image" src="https://github.com/user-attachments/assets/6a4342b0-21a5-4c78-ba5b-4e44148c17cf" />

    



