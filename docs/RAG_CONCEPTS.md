# RAG & Embeddings for Lovli

This document explains the core concepts behind the Lovli legal search engine.

## What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique to give an LLM access to specific, up-to-date data (like Norwegian laws) without retraining the model.

1. **Retrieve**: When a user asks a question, we search our database for the most relevant law paragraphs.
2. **Augment**: We add those paragraphs to the LLM's prompt as context.
3. **Generate**: The LLM answers the question using only the provided legal text.

## How Embeddings Work
Embeddings are the "mathematical DNA" of text. They convert words and sentences into long lists of numbers (vectors).

- **Semantic Search**: Unlike keyword search (which looks for exact words), embedding search looks for *meaning*.
- **Example**: A search for "oppsigelse av leilighet" (terminating an apartment lease) will find laws about "leieforholdets opphør" (end of tenancy) because their vectors are mathematically close.

## The Lovli Pipeline
1. **XML Parsing**: We read the Lovdata XML files and extract each legal article (`<article>`).
2. **Vectorization**: We send the text of each article to the **BGE-M3** model to get its embedding vector.
3. **Storage**: We store the text and its vector in **Qdrant**.
4. **Querying**:
   - User asks: "Can I get my deposit back?"
   - We vectorize the question.
   - Qdrant finds the top 3-5 most similar law articles.
   - LLM (GLM-4.7) reads those articles and explains the answer in plain Norwegian.

## Citations & Accuracy
Because we use RAG, every answer can be linked back to a specific `data-lovdata-URL` found in the source XML, ensuring users can verify the information at the source.
