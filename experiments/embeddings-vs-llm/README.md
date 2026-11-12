# Embeddings vs LLMs: Why Does RAG Need Both?

I wanted to understand what the embedding model and the LLM are actually doing inside a RAG flow.

For this experiment, I used a small support knowledge base and kept the retrieval and generation steps separate.

The setup uses:

- `nomic-embed-text` for embeddings
- `llama3.2:3b` for answer generation
- Ollama to run both models locally

## The question

I used this question:

> I entered the wrong password several times and now I can't sign in.

The knowledge base didn't contain the same wording.

The relevant support case was:

> Accounts are temporarily locked after five failed login attempts. The account unlocks automatically after 30 minutes.

## Finding the relevant information

I converted the question and support cases into embeddings using `nomic-embed-text`.

Then I compared the question embedding with each support case using cosine similarity.

The relevant account-lockout case was retrieved with a similarity score of `0.649`.

This step didn't generate an answer. It only helped find the information that was closest to the user's question.

## Generating the answer

I then passed the retrieved support case to `llama3.2:3b` along with the original question.

The generated answer was:

> Your account is now temporarily locked due to the five failed login attempts. Don't worry, it will unlock automatically after 30 minutes.

So the flow was:

Question → Embedding → Similarity Search → Relevant Context → LLM → Answer

## What happens with the wrong context?

I also gave the LLM an unrelated support case while keeping the question the same.

This helped show that retrieval quality matters. If the retrieval step brings back the wrong information, the LLM doesn't have the right knowledge-base context to work with.

## LLM alone vs retrieval + LLM

The LLM can still answer some questions directly using what it already knows.

But that's different from answering using information from our own knowledge base.

For example, if an internal support rule says an account unlocks after a specific amount of time, retrieval gives the LLM access to that information instead of relying only on what the model already knows.

## What I took from this

The two models are doing different jobs.

The embedding model helps find the relevant context.

The LLM uses that context to generate the answer.

**Embeddings help find the context. The LLM uses that context to generate the answer.**

The full experiment is available in [`experiment.ipynb`](./experiment.ipynb).

## Running the experiment

This experiment uses Ollama to run the embedding model and LLM locally.

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```
