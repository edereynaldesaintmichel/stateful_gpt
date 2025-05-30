# Stateful GPT: A (Partially) Failed Attempt at Improving the Decoder-Only Transformer Architecture

## Problem

When a transformer generates a token, the last hidden state's information (n_embd * 32 (or 16) bits) is collapsed into a single token (log2(vocab_size) bits). This seems suboptimal, as valuable information is lost in the compression.

## Proposed Solution

Enrich token embeddings with the last hidden state of the inference step that generated them.

I used a custom architecture, similar to attention and LSTM input gates:

*   **Component-wise key-query attention:** Token embeddings act as queries, and a `nn.Linear(hidden_states)` projection provides the keys.
*   Another `nn.Linear(hidden_states)` projection generates the values.
*   The values are added to the token embeddings, weighted by the component-wise attention score.

## Theoretical Challenge

This architecture is fundamentally recurrent, which theoretically hinders parallelization during training.

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

$CE_{loss}(T_n) < CE_{loss}(T_0)$ for every recurrence depth n during training. Otherwise, the model would just learn to not use any hidden state's information. Even though it doesn't formally imply that this inequality holds even for recurrence depth not seen during training, it is empirically observed.

## Practical observations

This architecture, although theoretically promising, is only useful for very shallow transformers. Indeed, the parallelized training performs multiple backpropagation steps consecutively on each batch of data, which necessitates do dial down the learning rate for recurrence training. This implies that, even though the stateful transformer far exceeds the performance of a vanilla transformer in terms of performance vs. number of training epochs, it lags behind when the compute intensity of each epoch is taken into account.

It should be noted, though, that even accounting for the compute overhead, the stateful transfomrer outperforms the base architecture for very shallow transformers. Hence the "**partial**" failure mentioned above. You can test this by setting n_layers = 2 in the stateful_gpt.py training script.

Also, for a fully trained Stateful Transformer, Norm(enrichment) ~ Norm(token_embeddings) * 0.2, meaning that some information indeed flows through the recurrent connection. It's just not worth the compute overhead, in most cases.
