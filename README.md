# Stateful GPT: A (Partially) Failed Attempt at Improving the Decoder-Only Transformer Architecture

## Problem

When a transformer generates a token, the last hidden state's information (n_embd * 32 (or 16) bits) is collapsed into a single token (log2(vocab_size) bits). This seems suboptimal, as valuable information is lost in the compression.

## Proposed Solution

Enrich token embeddings with the last hidden state of the inference step that generated them.

I used a custom architecture, similar to attention and LSTM's input gates:

*   **Component-wise key-query attention:** Token embeddings act as queries, and `nn.Linear(hidden_states)` provides the keys.
*   `nn.Linear(hidden_states)` generates the values.
*   The values are added to the token embeddings, weighted by the component-wise attention score.

## Theoretical Challenge

This architecture is fundamentally recurrent, which theoretically hinders parallelization during training.  This is a significant drawback for Transformers.

## Proposed Solution: Parallel Training for a Recurrent Architecture

To mitigate the parallelization issue, a parallel training technique was devised:

1.  **Parallel Training Step:** A standard, fully parallelized training step is performed on the transformer. No information flows through the recurrent connection at this point. Cross-entropy loss (CE-loss) is computed, and normal backpropagation is performed. The last hidden states are stored.
2.  **Recurrent Training Step:** The last hidden states from the previous step are fed into the model to enrich the token embeddings. Backpropagation is performed as usual, and new hidden states are stored.
3.  **Iterative Recurrence:** The second step can be repeated indefinitely, using the last step's hidden states, at the cost of additional training steps.

**Important Note:** Training and inference differ in their handling of recurrence depth:

*   **Training:** The recurrence depth is set to a fixed and finite number.
*   **Inference:** The effective recurrence degree is `generated_sequence_length - 1`.  The first token is generated without recurrence. Token 2 uses Token 1's hidden state (1 degree of recurrence), Token 3 uses Token 2's hidden state (2 degrees of recurrence), and so on.

The discrepancy between training and inference is addressed by ensuring stability during training. If the CE loss remains controlled with increased recurrence depth during training, the model should be stable even with theoretically infinite recurrence depth during inference.

More formally, training ensures that:

```
KL divergence(stateful_transformer_depth_of_n || stateful_transformer_depth_of_n_plus_one) < KL_divergence(stateful_transformer_depth_of_n_minus_one || stateful_transformer_depth_of_n)
```
```markdown
KL(P(x)||Q(x)) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
```

Where:

*   `KL` represents the Kullback-Leibler divergence.
*   `P(x)` and `Q(x)` represent probability distributions.
*   `X` is the set of all possible values of x.

```
KL divergence(stateful_transformer_depth_of_n || stateful_transformer_depth_of_n_plus_one) < KL_divergence(stateful_transformer_depth_of_n_minus_one || stateful_transformer_depth_of_n)
```

This inequality means that the KL divergence between the output distributions of the stateful transformer at recurrence depth `n` and `n+1` is smaller than the KL divergence between the output distributions at depths `n-1` and `n`. This suggests that the output distributions become more similar as the recurrence depth increases, indicating stability.
