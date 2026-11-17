# Does Chunk Size Change What RAG Retrieves?

I wanted to see how much chunk size affects retrieval in a RAG system.

For this experiment, I used the same support document and split it in three different ways:

- Small chunks — mostly one sentence at a time
- Medium chunks — related sentences kept together
- Large chunks — complete sections kept together

Everything else stayed the same. I used `nomic-embed-text` for embeddings and cosine similarity for retrieval.

## Test 1 — Account lock

Question:

> My account got locked after entering the wrong password. How long do I need to wait?

### Retrieval results

| Chunk size | Similarity | Result |
|---|---:|---|
| Small | 0.740 | Returned only the 30-minute unlock information |
| Medium | 0.781 | Returned the lock reason, unlock time, and related guidance |
| Large | 0.791 | Returned the answer along with additional account-access information |

All three contained enough information to answer the question.

The generated answers were also very similar. In this case, chunk size didn't make much difference to the final answer.

## Test 2 — Password reset

Question:

> My password reset link expired. How long was it valid, and what should I do now?

### Retrieval results

| Chunk size | Similarity | Result |
|---|---:|---|
| Small | 0.801 | Contained only the 15-minute validity |
| Medium | 0.809 | Contained the validity and what to do after expiry |
| Large | 0.768 | Contained both facts along with other account-access information |

This question needed two related pieces of information.

The small chunk was highly relevant, but it didn't contain the complete answer. The medium chunk kept both required facts together.

Interestingly, the LLM still filled in the missing action when given the small chunk. That information came from the model itself, not from the retrieved context.

## Test 3 — Two-factor authentication

Question:

> I want to enable two-factor authentication. Where do I enable it, and what will I need when I sign in afterward?

### Retrieval results

| Chunk size | Similarity | Result |
|---|---:|---|
| Small | 0.850 | Contained where to enable 2FA, but not what happens during sign-in |
| Medium | 0.845 | Contained both required facts |
| Large | 0.803 | Contained both facts along with recovery-code information |

This was another case where the small chunk had the highest similarity score but didn't contain everything needed to answer the full question.

## What I found

The three tests showed different tradeoffs.

Small chunks were focused and sometimes had the highest similarity, but related information could get separated.

Medium chunks kept related facts together without bringing in much unrelated information.

Large chunks usually contained enough information, but sometimes included details that weren't needed for the question.

For these examples, medium chunks gave a useful balance between focused retrieval and enough surrounding context.

## Takeaway

A higher similarity score doesn't automatically mean the retrieved context is better for answering the full question.

The chunk also needs to contain enough information for the task.

Too small can separate related information. Too large can bring unnecessary information along.

So instead of asking:

> What is the best chunk size?

A more useful question is:

> Does the retrieved chunk contain the right amount of context for the question?

This is a small controlled experiment, so the results shouldn't be treated as a general benchmark for chunk sizes.

The full experiment is available in [`experiment.ipynb`](./experiment.ipynb).