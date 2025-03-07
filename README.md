# Stateful GPT: a (partially) failed attempt at improving the Decoder-Only Transformer architecture

Problem: when a transformer generates a token, the last hidden state's information (n_embd * 32 (or 16) bits) is collapsed into a single token (log2(vocab_size) bits). I thought this was suboptimal.

### Proposed solution: enriching the token embeddings with the last hidden state of the inference step that generated them. 
I used a custom architecture, similar to attention and LSTM's input gates: component-wise key-query attention, with token-embeddings being the queries and nn.Linear(hidden_states) being the keys. Another nn.Linear(hidden_states) being the values, which are added to the token embeddings in proportion of the aforementioned component-wise attention score.

### Theoretical challenge: this architecture is basically a RNN, which theoretically can't leverage parallelization during training. And that sucks.
Proposed solution: Even though this architecture is fully recurrent, a parallel training technique was devised: 
- First, a normal, fully parallelized training step is conducted on the transformer. No information flows through the recurrent connection at this point. CE-loss is computed and normal backprop is performed. Last hidden states are stored.
- Second, a recurrent training step is performed: the first step's last hidden states are fed into the model and enrich token embeddings. Backprop is performed the same. New hidden states are stored.
- The second step can be repeated indefinitely using the last step's hidden states, at the cost of an additional training step.

Please note that in this case, training and inference are performed differently: During training, the recurrence depth is set to a fixed and finite number. During inference, the effective recurrence degree is `generated_sequence_length - 1`. 
Indeed, the first token is generated without leveraging the recurrent nature of the network. Token 2 is generated using token 1's hidden sate (1 degree of recurrence). Token 3 is generated using token 2's hidden state, which is a 2nd order hidden state, etc...
However, this discrepancy between training and inference doesn't matter too much, as the stability condition during training ("CE loss doesn't get out of control with deeper recurrence") implies that 
