
h.0	= First transformer block
h.11 =	Twelfth (last) transformer block
attn =	Attention
c_attn =	Combined query, key, value projection
c_proj = under attn	Attention output projection
mlp =	Feed-forward neural network
c_fc =	First MLP linear layer (expansion)
c_proj = under mlp	Second MLP linear layer (compression)
ln_1 =	LayerNorm before attention
ln_2 =	LayerNorm before MLP
ln_f =	Final LayerNorm
wte =	Token embeddings
wpe =	Position embeddings
weight =	Learned matrix or scale vector
bias =	Learned offset vector, except special buffer names such as attn.bias