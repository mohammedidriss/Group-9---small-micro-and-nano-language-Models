Contributors:
Mohamad Idriss,
Sriram Acharya Mudumbai,
Vibin Chandrabose,
Walid El Mahdy,
Rohit Jacob Issac

### **Achieving High-Fidelity Output from Efficient Small Language Models**
 
**1 - Problem Statement** 

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, but their significant computational and financial costs present a major barrier to widespread adoption. Small Language Models (SLMs) offer a compelling alternative due to their efficiency, speed, and reduced resource requirements.
However, standard SLMs often suffer from a critical flaw: a lack of factual accuracy, leading to a high propensity for "hallucination" where the model generates plausible but incorrect information.This project aimed to investigate and implement methodologies to prove that an SLM can be engineered to be both computationally efficient and factually reliable, approaching the performance of an LLM with only a minor trade-off in accuracy.

**2 - Methodology and Iterative Scenarios**

We conducted a series of three iterative experiments to identify the optimal strategy for building a high-performance SLM.

**Scenario 1:** Baseline SLM with General-Purpose Fine-Tuning
- **Approach:** A small language model was trained from scratch using knowledge distillation from a larger teacher model (microsoft/phi-2). The training was performed on a small, general-purpose dataset (databricks/databricks-dolly-15k).
- **Results:** Failure. The model exhibited severe hallucination and topic drift, often providing repetitive and nonsensical answers.
- **Analysis:** The low-quality, unstructured dataset and insufficient training were inadequate to teach the model complex reasoning or instruction-following skills. The model learned basic language patterns but lacked any grounding in factual knowledge.

**Scenario 2:** SLM with Task-Specific Fine-Tuning and RAG

- **Approach:** Recognizing the need for a specific skill, we shifted our strategy. We fine-tuned a pre-trained SLM (distilgpt2) directly on the SQuAD (Stanford Question Answering Dataset). This dataset is explicitly designed to teach a model how to answer questions based on a given context. This trained model was then integrated into a Retrieval-Augmented Generation (RAG) pipeline.
- **Results:** Success. The model's accuracy improved dramatically. When provided with context retrieved from a knowledge base (Wikipedia), the model was able to generate correct, factually grounded answers.
- **Analysis:** Fine-tuning on a task-specific dataset successfully taught the model the crucial skill of contextual question-answering, which is the core requirement for an effective RAG system.

**Scenario 3:** Advanced SLM with QLoRA Fine-Tuning and RAG
- **Approach:** To further optimize efficiency, we replaced the standard fine-tuning method with QLoRA (Quantized Low-Rank Adaptation). This advanced technique allows for fine-tuning a larger base model (microsoft/phi-2) with significantly less memory by only training a small set of "adapter" layers. This QLoRA-tuned model was then integrated into the same RAG pipeline. 
- **Results:** Optimal Performance. This approach yielded the best results, demonstrating both high accuracy and exceptional computational efficiency during the training process. The model produced coherent, factually correct answers while benefiting from the memory savings of the QLoRA technique.
- **Analysis:** QLoRA proved to be the most effective method, allowing for the specialization of a more powerful base model than would have been possible with standard fine-tuning, leading to the highest quality outputs.


**4 - How to Run This Project** 

The use of QLoRA in the final scenario further demonstrates that these high-performance models can be created with exceptional computational efficiency, making advanced AI more accessible and sustainable.

This project is organized into a series of Google Colab cells. To replicate the results, run the cells in the provided notebook in order.

- Setup and Dependencies: Installs all necessary libraries and connects to your Google Drive. You must restart the Colab runtime after this cell completes.

* **For scenario 1:**  Fine-tunes the microsoft/phi-2 model on the databrick dataset and saves the final, trained model to your Google Drive for persistence.

* **For scenario 2:** Fine-tunes the distilgpt-2 model on the SQuAD dataset and saves the final, trained model to your Google Drive for persistence.

* **For scenario 3:** QLoRA Fine-Tuning: Fine-tunes the microsoft/phi-2 model on the SQuAD dataset and saves the final, trained model to your Google Drive for persistence.


- Build Vector Database: Creates the knowledge base for the RAG system. It processes Wikipedia articles, converts them to vector embeddings, builds a FAISS index for semantic search, and saves it to your Drive.

- for Scenarios 2 and 3: Interactive RAG Chat: Loads the fine-tuned model and vector database from your Drive to launch an interactive chat session where you can ask factual questions.

- Benchmark Evaluation: Runs the fine-tuned model against a suite of standardized academic benchmarks (MMLU, HellaSwag, etc.) to quantitatively measure its performance.

**The full code explanation:**
This comprehensive script is a complete end-to-end pipeline for fine-tuning a small language model (SLM), building a knowledge base for it to use, interacting with the model via a RAG (Retrieval-Augmented Generation) system, and finally, evaluating its performance with multiple benchmarks.

Here’s a breakdown of each major section:

### **1. Setup and Initialization**

  * **Install Libraries:** It begins by installing all the necessary Python packages. This includes `transformers` for the models, `datasets` for data, `bitsandbytes` for efficient model loading, and specialized libraries like `faiss-cpu` for vector search and `evaluate` for performance metrics.
  * **Mount Google Drive:** It connects to your Google Drive. This is a crucial step for a Colab notebook, as it allows the script to save large files (like the final model and the vector database) persistently.
  * **Import Libraries:** It imports all the required modules from the installed packages, making them ready for use.

### **2. Model Fine-Tuning**

This section takes a pre-trained small language model and further trains it on a specific task.

  * **Configuration:** It defines a `ProjectConfig` class to hold all important settings in one place, such as file paths, model names (`Microsoft\phi-2\`distilgpt2`), the dataset to use (`databricks`\`squad`), and training parameters like learning rate and batch size.
  * **Load Base Model:** It downloads "phi-2' for scenarion 1 & 3, `distilgpt2` for scenario 2, a smaller, distilled version of GPT-2, and its tokenizer. This model is chosen because it's small enough to train quickly.
   * **Prepare SQuAD Dataset:** It loads the Databricks\SQuAD. A custom function then formats this data into a specific "RAG-style" prompt, which looks like this:
    ```
    Context: [some paragraph]
    Instruction: [a question about the paragraph]
    Response: [the answer]
    ```
    Training the model on this format teaches it to answer questions based on a provided context, which is the core idea behind RAG.
  * ** Scenario 2 dataset** The databricks dataset was deliberitly selected in an attempt to prove that data quality is essential for the accuracy of the model.
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


****Results****

**Experiment 1**
**Model Name:** Microsoft phi-2

**Compression Ratio:** 94% reduction in size

**Dataset:** Databricks 15k

**Number of Paramters:** 82 Millions

**Inferencing Results:**

**Hallucination exists:** Yes

**Training Time:** 7 hours 12 mins (A100 GPU)

**Testing Results - NO RAG**
<img width="1108" height="384" alt="Screenshot 2025-10-09 at 12 29 59 PM" src="https://github.com/user-attachments/assets/4ba48509-d412-4001-89ac-9715374d95e7" />

** Benchmarks:**

<img width="589" height="262" alt="image" src="https://github.com/user-attachments/assets/83b10bb5-fa2f-4b8f-9de6-424ee8feba09" />



**Experiment 2**

**Model Name:** distilgpt-2

**Compression Ratio:** 94% reduction in size

**Dataset:** Standford questions and answers (SQUAD)

**Number of Paramters:** 82 Millions

**Inferencing Results:**

**Training Time:** 1.24 hours (A100 GPU)

**Hallucination exists:** No

 **Testing Results with RAG**
 
<img width="1108" height="384" alt="Screenshot 2025-10-09 at 12 29 59 PM" src="https://github.com/user-attachments/assets/4ded93bf-c603-4101-a370-8afd8b9a97dd" />

**Benchmarks**

<img width="347" height="173" alt="image" src="https://github.com/user-attachments/assets/2c3784ea-dbd0-4733-a07f-6dd0e6dc4f8f" />


    
**Experiment 3**

**Model Name:** Microsoft\phi-2

**Compression Ratio:** 94% reduction in size

**Dataset:** Standford questions and answers (SQUAD)

**Number of Paramters:** 82 Millions

**Inferencing Results:**

**Training Time:** 6 hours 53 mins (A100 GPU)

  * **Testing Results - QLORA with RAG:**
  * QLORA Model Performance
    
    <img width="509" height="82" alt="Screenshot 2025-10-08 at 6 34 13 PM" src="https://github.com/user-attachments/assets/e0cbd85d-d242-4018-b3d9-ceca1106eccc" />

    QlORA Model Training Results
    <img width="1138" height="523" alt="Screenshot 2025-10-08 at 6 32 51 PM" src="https://github.com/user-attachments/assets/fc22f39e-195a-47dc-b4b0-87076fd52bd3" />


**Conclusion** 

This project successfully demonstrates that SLMs can serve as a highly efficient and accurate alternative to their larger counterparts, provided they are engineered correctly.
The key finding is that a combination of two modern techniques is required:
 - **Data Quality is Paramount:** The failure of Scenario 1 and the success of Scenarios 2 and 3 prove that the quality, structure, and relevance of the training data are the most critical factors for success. General-purpose, unstructured datasets are insufficient for teaching complex, task-specific skills like contextual reasoning.

 - **Task-Specific Fine-Tuning:** The SLM must be explicitly trained on a high-quality dataset that matches the target task (e.g., SQuAD for RAG).

 - **Retrieval-Augmented Generation (RAG):** The model must be grounded in an external knowledge base to ensure factual accuracy and eliminate hallucination.

   

    



