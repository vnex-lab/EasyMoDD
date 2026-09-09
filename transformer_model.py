import os
from pathlib import Path

import numpy as np


def _configure_cuda_dll_search_path():
    if os.name != "nt":
        return
    candidates = []
    for variable in ("CUDA_PATH_V12_6", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value) / "bin")
    candidates.extend(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA").glob("v12.*\bin"))
    for directory in candidates:
        if directory.is_dir():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(directory))
            return


_configure_cuda_dll_search_path()

# GPU Support — same validated pattern as chatbot_model.py
try:
    import cupy as cp
    try:
        # Actually test GPU is usable before committing to it
        _test = cp.array([1, 2, 3])
        cp.random.seed(42)
        _test2 = cp.random.randn(4, 4)
        _ = cp.exp(_test2)
        _ = cp.dot(cp.ones((2, 2), dtype=cp.float32), cp.ones((2, 2), dtype=cp.float32))
        GPU_AVAILABLE = True
        print("Transformer: GPU verified - using CuPy!")
    except Exception as _e:
        print(f"Transformer: GPU not usable ({str(_e)[:60]}) - falling back to CPU")
        cp = np
        GPU_AVAILABLE = False
except ImportError:
    cp = np
    GPU_AVAILABLE = False

class TransformerChatbot:
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4,
                 ff_dim=1024, max_seq_len=50, learning_rate=0.003,
                 optimizer='adam', weight_decay=0.01,
                 scheduler='cosine', warmup_epochs=5,
                 dropout_rate=0.1, grad_clip=5.0):
        self.vocab_size   = vocab_size
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.num_layers   = num_layers
        self.ff_dim       = ff_dim
        self.max_seq_len  = max_seq_len
        self.learning_rate = learning_rate
        self.initial_lr   = learning_rate          # needed for cosine/warmup schedules
        self.current_epoch = 0
        self.gpu_available = GPU_AVAILABLE
        self.xp = cp if GPU_AVAILABLE else np

        # ---- Optimizer ----
        self.optimizer    = optimizer.lower()      # sgd | adam | adamw
        self.weight_decay = weight_decay           # L2 reg (adamw)
        self.grad_clip    = grad_clip
        self.opt_m        = {}                     # Adam 1st moment
        self.opt_v        = {}                     # Adam 2nd moment
        self.opt_t        = 0                      # Adam step counter

        # ---- Scheduler ----
        self.scheduler    = scheduler.lower()      # constant | linear | cosine | warmup_cosine | warmup_linear
        self.warmup_epochs = int(warmup_epochs)

        # ---- Regularisation ----
        self.dropout_rate = float(dropout_rate)    # 0.0 = off
        self.training     = True                   # False during generate()
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = embed_dim // num_heads
        
        # BETTER initialization for training from scratch!
        # Embeddings: larger scale for stronger signals
        self.embedding = self.xp.random.randn(vocab_size, embed_dim) * 0.1
        self.pos_encoding = self._create_positional_encoding()
        
        self.encoder_layers = []
        self.decoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(self._init_encoder_layer())
            self.decoder_layers.append(self._init_decoder_layer())
        
        # Xavier initialization for output layer
        scale = self.xp.sqrt(2.0 / (embed_dim + vocab_size))
        self.output_weights = self.xp.random.randn(embed_dim, vocab_size) * scale
        self.output_bias = self.xp.zeros((1, vocab_size))
        
        self.training_history = {'loss': []}
        
        print(f"Transformer Model Initialized!")
        print(f"   GPU: {'Enabled (CuPy)' if GPU_AVAILABLE else 'Disabled (NumPy)'}")
        print(f"   Vocab size: {vocab_size:,}")
        print(f"   Embed dim: {embed_dim}  |  Heads: {num_heads}  |  Layers: {num_layers}")
        print(f"   FF dim: {ff_dim}  |  Max seq len: {max_seq_len}")
        print(f"   Optimizer: {optimizer.upper()}  |  Scheduler: {scheduler}")
        print(f"   LR: {learning_rate}  |  Weight decay: {weight_decay}  |  Dropout: {dropout_rate}")
        print(f"   Warmup epochs: {warmup_epochs}  |  Grad clip: {grad_clip}")

        total_params = self._count_parameters()
        print(f"   📊 Total parameters: {total_params:,}")
        if total_params >= 1_000_000_000:
            print(f"   BILLION PARAMETER MODEL! {total_params/1e9:.2f}B params!")
        elif total_params >= 1_000_000:
            print(f"   {total_params/1e6:.1f}M parameter model")
    
    def _create_positional_encoding(self):
        position = self.xp.arange(self.max_seq_len)[:, self.xp.newaxis]
        div_term = self.xp.exp(self.xp.arange(0, self.embed_dim, 2) * 
                               -(self.xp.log(10000.0) / self.embed_dim))
        
        pos_encoding = self.xp.zeros((self.max_seq_len, self.embed_dim))
        pos_encoding[:, 0::2] = self.xp.sin(position * div_term)
        pos_encoding[:, 1::2] = self.xp.cos(position * div_term)
        
        return pos_encoding
    
    def _init_encoder_layer(self):
        # Xavier initialization for better gradient flow
        scale = self.xp.sqrt(2.0 / self.embed_dim)
        scale_ff = self.xp.sqrt(2.0 / (self.embed_dim + self.ff_dim))
        
        layer = {
            'Wq': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'ff_w1': self.xp.random.randn(self.embed_dim, self.ff_dim) * scale_ff,
            'ff_b1': self.xp.zeros((1, self.ff_dim)),
            'ff_w2': self.xp.random.randn(self.ff_dim, self.embed_dim) * scale_ff,
            'ff_b2': self.xp.zeros((1, self.embed_dim)),
            'ln1_gamma': self.xp.ones((1, self.embed_dim)),
            'ln1_beta': self.xp.zeros((1, self.embed_dim)),
            'ln2_gamma': self.xp.ones((1, self.embed_dim)),
            'ln2_beta': self.xp.zeros((1, self.embed_dim)),
        }
        return layer
    
    def _init_decoder_layer(self):
        # Xavier initialization for better gradient flow
        scale = self.xp.sqrt(2.0 / self.embed_dim)
        scale_ff = self.xp.sqrt(2.0 / (self.embed_dim + self.ff_dim))
        
        layer = {
            'Wq_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo_self': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wq_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wk_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wv_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'Wo_cross': self.xp.random.randn(self.embed_dim, self.embed_dim) * scale,
            'ff_w1': self.xp.random.randn(self.embed_dim, self.ff_dim) * scale_ff,
            'ff_b1': self.xp.zeros((1, self.ff_dim)),
            'ff_w2': self.xp.random.randn(self.ff_dim, self.embed_dim) * scale_ff,
            'ff_b2': self.xp.zeros((1, self.embed_dim)),
            'ln1_gamma': self.xp.ones((1, self.embed_dim)),
            'ln1_beta': self.xp.zeros((1, self.embed_dim)),
            'ln2_gamma': self.xp.ones((1, self.embed_dim)),
            'ln2_beta': self.xp.zeros((1, self.embed_dim)),
            'ln3_gamma': self.xp.ones((1, self.embed_dim)),
            'ln3_beta': self.xp.zeros((1, self.embed_dim)),
        }
        return layer
    
    def _count_parameters(self):
        count = self.embedding.size + self.output_weights.size + self.output_bias.size
        
        for layer in self.encoder_layers:
            for param in layer.values():
                count += param.size
        
        for layer in self.decoder_layers:
            for param in layer.values():
                count += param.size
        
        return count
    
    def _layer_norm(self, x, gamma, beta, epsilon=1e-5):
        mean = self.xp.mean(x, axis=-1, keepdims=True)
        var = self.xp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / self.xp.sqrt(var + epsilon)
        return gamma * x_norm + beta
    
    def _multi_head_attention(self, Q, K, V, Wq, Wk, Wv, Wo, mask=None):
        batch_size, q_seq_len, _ = Q.shape
        _, k_seq_len, _ = K.shape
        _, v_seq_len, _ = V.shape
        
        Q_proj = self.xp.dot(Q, Wq)
        K_proj = self.xp.dot(K, Wk)
        V_proj = self.xp.dot(V, Wv)
        
        Q_heads = Q_proj.reshape(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K_heads = K_proj.reshape(batch_size, k_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V_heads = V_proj.reshape(batch_size, v_seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        scores = self.xp.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / self.xp.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = self._softmax(scores, axis=-1)
        
        attn_output = self.xp.matmul(attn_weights, V_heads)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_seq_len, self.embed_dim)
        
        output = self.xp.dot(attn_output, Wo)
        
        return output, attn_weights
    
    def _feed_forward(self, x, w1, b1, w2, b2):
        hidden = self.xp.maximum(0, self.xp.dot(x, w1) + b1)
        output = self.xp.dot(hidden, w2) + b2
        return output
    
    def _softmax(self, x, axis=-1):
        # Subtract max for stability
        x_max = self.xp.max(x, axis=axis, keepdims=True)
        exp_x = self.xp.exp(x - x_max)
        # Add epsilon to denominator
        sum_exp = self.xp.sum(exp_x, axis=axis, keepdims=True) + 1e-10
        return exp_x / sum_exp
    
    def _create_causal_mask(self, seq_len):
        mask = self.xp.triu(self.xp.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9
        return mask
    
    def encode(self, input_seq):
        batch_size, seq_len = input_seq.shape
        
        x = self.embedding[input_seq]
        x = x + self.pos_encoding[:seq_len]
        
        for layer in self.encoder_layers:
            attn_output, _ = self._multi_head_attention(
                x, x, x, 
                layer['Wq'], layer['Wk'], layer['Wv'], layer['Wo']
            )
            x = self._layer_norm(x + attn_output, layer['ln1_gamma'], layer['ln1_beta'])
            
            ff_output = self._feed_forward(
                x, layer['ff_w1'], layer['ff_b1'], layer['ff_w2'], layer['ff_b2']
            )
            x = self._layer_norm(x + ff_output, layer['ln2_gamma'], layer['ln2_beta'])
        
        return x
    
    def decode(self, target_seq, encoder_output):
        batch_size, target_len = target_seq.shape
        
        x = self.embedding[target_seq]
        x = x + self.pos_encoding[:target_len]
        
        causal_mask = self._create_causal_mask(target_len)
        
        for layer in self.decoder_layers:
            self_attn_output, _ = self._multi_head_attention(
                x, x, x,
                layer['Wq_self'], layer['Wk_self'], layer['Wv_self'], layer['Wo_self'],
                mask=causal_mask
            )
            x = self._layer_norm(x + self_attn_output, layer['ln1_gamma'], layer['ln1_beta'])
            
            cross_attn_output, _ = self._multi_head_attention(
                x, encoder_output, encoder_output,
                layer['Wq_cross'], layer['Wk_cross'], layer['Wv_cross'], layer['Wo_cross']
            )
            x = self._layer_norm(x + cross_attn_output, layer['ln2_gamma'], layer['ln2_beta'])
            
            ff_output = self._feed_forward(
                x, layer['ff_w1'], layer['ff_b1'], layer['ff_w2'], layer['ff_b2']
            )
            x = self._layer_norm(x + ff_output, layer['ln3_gamma'], layer['ln3_beta'])
        
        return x
    
    def forward(self, input_seq, target_seq):
        encoder_output = self.encode(input_seq)
        decoder_output = self.decode(target_seq, encoder_output)
        
        logits = self.xp.dot(decoder_output, self.output_weights) + self.output_bias
        
        return logits
    
    def train_batch(self, input_batch, target_batch, learning_rate=None):
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        batch_size = len(input_batch)
        total_loss = 0.0
        
        for input_seq, target_seq in zip(input_batch, target_batch):
            input_seq = self.xp.array(input_seq).reshape(1, -1)
            target_input = self.xp.array(target_seq[:-1]).reshape(1, -1)
            target_output = self.xp.array(target_seq[1:]).reshape(1, -1)
            
            logits = self.forward(input_seq, target_input)
            
            probs = self._softmax(logits, axis=-1)
            
            loss = 0.0
            for t in range(target_output.shape[1]):
                correct_token = int(target_output[0, t])
                prob = float(probs[0, t, correct_token])
                loss += -self.xp.log(prob + 1e-10)
            
            loss /= target_output.shape[1]
            total_loss += float(loss)
            
            self._backward_and_update(input_seq, target_input, target_output, probs, learning_rate)
        
        return total_loss / batch_size

    def _opt_update(self, param, grad, param_id, lr_scale=1.0):
        """Apply SGD / Adam / AdamW update in-place on `param`."""
        xp = self.xp
        effective_lr = self.learning_rate * lr_scale
        clipped = xp.clip(grad, -self.grad_clip, self.grad_clip)

        if self.optimizer == 'sgd':
            param -= effective_lr * clipped
            return

        # ---- Adam / AdamW ----
        if param_id not in self.opt_m:
            self.opt_m[param_id] = xp.zeros_like(param)
            self.opt_v[param_id] = xp.zeros_like(param)

        b1, b2, eps = 0.9, 0.999, 1e-8
        self.opt_m[param_id] = b1 * self.opt_m[param_id] + (1 - b1) * clipped
        self.opt_v[param_id] = b2 * self.opt_v[param_id] + (1 - b2) * clipped ** 2

        t = max(self.opt_t, 1)
        m_hat = self.opt_m[param_id] / (1.0 - b1 ** t)
        v_hat = self.opt_v[param_id] / (1.0 - b2 ** t)
        step  = effective_lr * m_hat / (xp.sqrt(v_hat) + eps)

        if self.optimizer == 'adamw':
            param -= step + effective_lr * self.weight_decay * param
        else:
            param -= step

    def step_lr(self, total_epochs=100):
        """Update learning rate once per epoch according to the chosen scheduler."""
        import math
        self.current_epoch += 1
        epoch = self.current_epoch
        lr0   = self.initial_lr
        sched = self.scheduler

        if sched == 'constant':
            self.learning_rate = lr0

        elif sched == 'linear':
            if self.learning_rate > 1e-6:
                self.learning_rate *= 0.99

        elif sched == 'cosine':
            progress = min(epoch / max(total_epochs, 1), 1.0)
            self.learning_rate = max(lr0 * 0.5 * (1.0 + math.cos(math.pi * progress)),
                                     lr0 * 0.01)

        elif sched == 'warmup_cosine':
            if epoch <= self.warmup_epochs:
                self.learning_rate = lr0 * epoch / max(self.warmup_epochs, 1)
            else:
                rest     = max(total_epochs - self.warmup_epochs, 1)
                progress = (epoch - self.warmup_epochs) / rest
                self.learning_rate = max(lr0 * 0.5 * (1.0 + math.cos(math.pi * progress)),
                                         lr0 * 0.01)

        elif sched == 'warmup_linear':
            if epoch <= self.warmup_epochs:
                self.learning_rate = lr0 * epoch / max(self.warmup_epochs, 1)
            else:
                rest     = max(total_epochs - self.warmup_epochs, 1)
                progress = (epoch - self.warmup_epochs) / rest
                self.learning_rate = max(lr0 * (1.0 - progress), lr0 * 0.01)
    
    def _attn_forward_cached(self, Q, K, V, Wq, Wk, Wv, Wo, mask=None):
        """Multi-head attention — returns output + full cache for Wq/Wk/Wv/Wo backward."""
        xp = self.xp
        batch_size, q_len, _ = Q.shape
        _, k_len, _ = K.shape

        Q_proj = xp.dot(Q, Wq)
        K_proj = xp.dot(K, Wk)
        V_proj = xp.dot(V, Wv)

        Q_h = Q_proj.reshape(batch_size, q_len,  self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K_h = K_proj.reshape(batch_size, k_len,  self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V_h = V_proj.reshape(batch_size, k_len,  self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = xp.matmul(Q_h, K_h.transpose(0, 1, 3, 2)) / xp.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn_w = self._softmax(scores, axis=-1)

        ctx_h = xp.matmul(attn_w, V_h)                                           # (B, H, q_len, head_dim)
        ctx   = ctx_h.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.embed_dim)  # pre-Wo
        output = xp.dot(ctx, Wo)

        # Full cache for backward: Q/K/V inputs + all projections + attention weights
        cache = {'Q': Q, 'K': K, 'V': V,
                 'Q_h': Q_h, 'K_h': K_h, 'V_h': V_h,
                 'attn_w': attn_w, 'ctx': ctx}
        return output, cache

    def _attn_backward(self, d_out, cache, layer_Wq, layer_Wk, layer_Wv, layer_Wo, lr,
                        is_self_attn=False, attn_id='attn'):
        """Full multi-head attention backward. Trains Wq, Wk, Wv, Wo in-place.

        is_self_attn=True: Q=K=V come from the same x, so gradient = d_Q + d_K + d_V.
        is_self_attn=False (cross-attn): K/V come from encoder — only d_Q returned.
        """
        xp = self.xp
        Q, K, V   = cache['Q'],  cache['K'],  cache['V']
        Q_h, K_h, V_h = cache['Q_h'], cache['K_h'], cache['V_h']
        attn_w, ctx   = cache['attn_w'], cache['ctx']
        B, q_len, _ = Q.shape
        _, k_len, _ = K.shape

        # ---- compute ALL gradients before any weight update ----

        # Wo gradient
        d_Wo = xp.dot(ctx.reshape(-1, self.embed_dim).T,
                       d_out.reshape(-1, self.embed_dim))

        # gradient through Wo → ctx (original Wo, not yet updated)
        d_ctx   = xp.dot(d_out, layer_Wo.T)
        d_ctx_h = d_ctx.reshape(B, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # V backward: ctx_h = attn_w @ V_h
        d_V_h    = xp.matmul(attn_w.transpose(0, 1, 3, 2), d_ctx_h)
        d_attn_w = xp.matmul(d_ctx_h, V_h.transpose(0, 1, 3, 2))

        # softmax backward
        d_scores = attn_w * (d_attn_w - xp.sum(d_attn_w * attn_w, axis=-1, keepdims=True))
        d_scores = d_scores / xp.sqrt(self.head_dim)

        # Q, K backward: scores = Q_h @ K_h.T
        d_Q_h = xp.matmul(d_scores,                    K_h)
        d_K_h = xp.matmul(d_scores.transpose(0,1,3,2), Q_h)

        # reshape to (B, seq_len, embed_dim)
        d_Q_proj = d_Q_h.transpose(0,2,1,3).reshape(B, q_len, self.embed_dim)
        d_K_proj = d_K_h.transpose(0,2,1,3).reshape(B, k_len, self.embed_dim)
        d_V_proj = d_V_h.transpose(0,2,1,3).reshape(B, k_len, self.embed_dim)

        # Wq, Wk, Wv gradients
        d_Wq = xp.dot(Q.reshape(-1, self.embed_dim).T, d_Q_proj.reshape(-1, self.embed_dim))
        d_Wk = xp.dot(K.reshape(-1, self.embed_dim).T, d_K_proj.reshape(-1, self.embed_dim))
        d_Wv = xp.dot(V.reshape(-1, self.embed_dim).T, d_V_proj.reshape(-1, self.embed_dim))

        # gradient back to input x — use original (unmodified) weight matrices
        d_Q_in = xp.dot(d_Q_proj.reshape(-1, self.embed_dim),
                          layer_Wq.T).reshape(B, q_len, self.embed_dim)
        if is_self_attn:
            # Q=K=V=x — full gradient = (d_Q + d_K + d_V) / 3  (divide to keep magnitude stable)
            d_K_in = xp.dot(d_K_proj.reshape(-1, self.embed_dim),
                              layer_Wk.T).reshape(B, k_len, self.embed_dim)
            d_V_in = xp.dot(d_V_proj.reshape(-1, self.embed_dim),
                              layer_Wv.T).reshape(B, k_len, self.embed_dim)
            d_input = (d_Q_in + d_K_in + d_V_in) / 3.0
        else:
            # cross-attention: K, V come from encoder — only Q gradient flows back to decoder
            d_input = d_Q_in

        # ---- apply all weight updates via optimizer ----
        # Wq/Wk: gradient through softmax is noisy → lr_scale 0.1
        # Wv: more direct value path → lr_scale 0.2; Wo: approx gradient → lr_scale 0.3
        self._opt_update(layer_Wo, d_Wo, f'{attn_id}_Wo', lr_scale=0.3)
        self._opt_update(layer_Wv, d_Wv, f'{attn_id}_Wv', lr_scale=0.2)
        self._opt_update(layer_Wq, d_Wq, f'{attn_id}_Wq', lr_scale=0.1)
        self._opt_update(layer_Wk, d_Wk, f'{attn_id}_Wk', lr_scale=0.1)

        return d_input

    def _apply_dropout(self, x):
        """Inverted dropout: scale kept activations up so inference needs no scaling."""
        if self.dropout_rate <= 0.0 or not self.training:
            return x, None
        xp = self.xp
        mask = (xp.random.random(x.shape) > self.dropout_rate).astype(x.dtype)
        return x * mask / (1.0 - self.dropout_rate), mask

    def _encode_with_cache(self, input_seq):
        """Forward pass through encoder, capturing all intermediates for full backward."""
        xp = self.xp
        batch_size, seq_len = input_seq.shape
        x = self.embedding[input_seq] + self.pos_encoding[:seq_len]
        cache = []
        for layer in self.encoder_layers:
            attn_out, attn_cache = self._attn_forward_cached(x, x, x,
                layer['Wq'], layer['Wk'], layer['Wv'], layer['Wo'])
            x = self._layer_norm(x + attn_out, layer['ln1_gamma'], layer['ln1_beta'])
            ff_input = x
            ff_pre   = xp.dot(x, layer['ff_w1']) + layer['ff_b1']
            ff_relu_raw = xp.maximum(0, ff_pre)
            ff_relu, drop_mask = self._apply_dropout(ff_relu_raw)
            ff_out   = xp.dot(ff_relu, layer['ff_w2']) + layer['ff_b2']
            x = self._layer_norm(x + ff_out, layer['ln2_gamma'], layer['ln2_beta'])
            cache.append({'ff_input': ff_input, 'ff_pre': ff_pre, 'ff_relu': ff_relu,
                          'drop_mask': drop_mask, 'self_attn': attn_cache})
        return x, cache

    def _decode_with_cache(self, target_input, encoder_output):
        """Forward pass through decoder, capturing all intermediates for full backward."""
        xp = self.xp
        batch_size, target_len = target_input.shape
        x = self.embedding[target_input] + self.pos_encoding[:target_len]
        causal_mask = self._create_causal_mask(target_len)
        cache = []
        for layer in self.decoder_layers:
            self_out, self_attn_cache = self._attn_forward_cached(x, x, x,
                layer['Wq_self'], layer['Wk_self'], layer['Wv_self'], layer['Wo_self'],
                mask=causal_mask)
            x = self._layer_norm(x + self_out, layer['ln1_gamma'], layer['ln1_beta'])

            cross_out, cross_attn_cache = self._attn_forward_cached(x, encoder_output, encoder_output,
                layer['Wq_cross'], layer['Wk_cross'], layer['Wv_cross'], layer['Wo_cross'])
            ff_input = self._layer_norm(x + cross_out, layer['ln2_gamma'], layer['ln2_beta'])

            ff_pre  = xp.dot(ff_input, layer['ff_w1']) + layer['ff_b1']
            ff_relu_raw = xp.maximum(0, ff_pre)
            ff_relu, drop_mask = self._apply_dropout(ff_relu_raw)
            ff_out  = xp.dot(ff_relu, layer['ff_w2']) + layer['ff_b2']
            x = self._layer_norm(ff_input + ff_out, layer['ln3_gamma'], layer['ln3_beta'])
            cache.append({'ff_input': ff_input, 'ff_pre': ff_pre, 'ff_relu': ff_relu,
                          'drop_mask': drop_mask,
                          'self_attn': self_attn_cache, 'cross_attn': cross_attn_cache})
        return x, cache

    def _ff_backward(self, d_x_out, layer, cache, layer_id):
        """Backprop through one FF sub-layer. Updates weights. Returns gradient w.r.t FF input."""
        xp = self.xp
        ff_input   = cache['ff_input']
        ff_pre     = cache['ff_pre']
        ff_relu    = cache['ff_relu']          # post-dropout value
        drop_mask  = cache.get('drop_mask')
        target_len = ff_input.shape[1]

        d_ff_out = d_x_out

        # ---- ff_w2 backward ----
        d_W2 = xp.dot(ff_relu.reshape(-1, self.ff_dim).T,
                       d_ff_out.reshape(-1, self.embed_dim))
        d_b2 = xp.sum(d_ff_out.reshape(-1, self.embed_dim), axis=0, keepdims=True)
        d_ff_relu_drop = xp.dot(d_ff_out.reshape(-1, self.embed_dim),
                                 layer['ff_w2'].T).reshape(1, target_len, self.ff_dim)

        # ---- undo dropout ----
        if drop_mask is not None:
            d_ff_relu = d_ff_relu_drop * drop_mask / (1.0 - self.dropout_rate)
        else:
            d_ff_relu = d_ff_relu_drop

        # ---- undo ReLU ----
        d_ff_pre = d_ff_relu * (ff_pre > 0)

        # ---- ff_w1 backward ----
        d_W1 = xp.dot(ff_input.reshape(-1, self.embed_dim).T,
                       d_ff_pre.reshape(-1, self.ff_dim))
        d_b1 = xp.sum(d_ff_pre.reshape(-1, self.ff_dim), axis=0, keepdims=True)

        # Gradient back through ff_input (for next layer's output)
        d_ff_input = xp.dot(d_ff_pre.reshape(-1, self.ff_dim),
                             layer['ff_w1'].T).reshape(ff_input.shape)

        # Update FF weights via optimizer
        self._opt_update(layer['ff_w1'], d_W1, f'{layer_id}_ff_w1')
        self._opt_update(layer['ff_b1'], d_b1, f'{layer_id}_ff_b1')
        self._opt_update(layer['ff_w2'], d_W2, f'{layer_id}_ff_w2')
        self._opt_update(layer['ff_b2'], d_b2, f'{layer_id}_ff_b2')

        # Total gradient back through residual + FF input
        return d_x_out + d_ff_input

    def _backward_and_update(self, input_seq, target_input, target_output, probs, lr):
        xp = self.xp
        batch_size, target_len, _ = probs.shape

        # === CROSS-ENTROPY GRADIENT ===
        d_logits = probs.copy()
        for t in range(target_len):
            d_logits[0, t, int(target_output[0, t])] -= 1.0
        d_logits /= max(target_len, 1)

        # === FORWARD WITH CACHE (captures all intermediates for full backward) ===
        encoder_output, enc_cache = self._encode_with_cache(input_seq)
        decoder_output, dec_cache = self._decode_with_cache(target_input, encoder_output)

        # === OUTPUT LAYER ===
        d_W_out = xp.dot(decoder_output.reshape(-1, self.embed_dim).T,
                          d_logits.reshape(-1, self.vocab_size))
        d_b_out = xp.sum(d_logits.reshape(-1, self.vocab_size), axis=0, keepdims=True)
        self._opt_update(self.output_weights, d_W_out, 'out_W')
        self._opt_update(self.output_bias,    d_b_out, 'out_b')

        # Gradient into the last decoder layer's output (use original output_weights before update)
        d_x = xp.dot(d_logits, self.output_weights.T)

        d_embed = xp.zeros_like(self.embedding)

        # Increment Adam step counter once per forward-backward pass
        self.opt_t += 1

        # === DECODER LAYERS BACKWARD (deepest to shallowest) ===
        n = self.num_layers
        for i, (layer, cache) in enumerate(zip(reversed(self.decoder_layers), reversed(dec_cache))):
            li = n - 1 - i   # actual layer index (for unique param IDs)
            d_x = self._ff_backward(d_x, layer, cache, f'dec{li}')
            d_x = self._attn_backward(d_x, cache['cross_attn'],
                                       layer['Wq_cross'], layer['Wk_cross'],
                                       layer['Wv_cross'], layer['Wo_cross'], lr,
                                       is_self_attn=False, attn_id=f'dec{li}_cross')
            d_x = self._attn_backward(d_x, cache['self_attn'],
                                       layer['Wq_self'], layer['Wk_self'],
                                       layer['Wv_self'], layer['Wo_self'], lr,
                                       is_self_attn=True, attn_id=f'dec{li}_self')

        # Gradient to target embeddings
        for t in range(target_input.shape[1]):
            tok = int(target_input[0, t])
            if tok < self.vocab_size:
                d_embed[tok] += d_x[0, t, :]

        # === ENCODER LAYERS BACKWARD ===
        enc_seq_len = input_seq.shape[1]
        d_enc = xp.broadcast_to(
            xp.mean(d_x, axis=1, keepdims=True),
            (1, enc_seq_len, self.embed_dim)
        ).copy()

        for i, (layer, cache) in enumerate(zip(reversed(self.encoder_layers), reversed(enc_cache))):
            li = n - 1 - i
            d_enc = self._ff_backward(d_enc, layer, cache, f'enc{li}')
            d_enc = self._attn_backward(d_enc, cache['self_attn'],
                                         layer['Wq'], layer['Wk'],
                                         layer['Wv'], layer['Wo'], lr,
                                         is_self_attn=True, attn_id=f'enc{li}_self')

        # Gradient to input embeddings
        for t in range(input_seq.shape[1]):
            tok = int(input_seq[0, t])
            if tok < self.vocab_size:
                d_embed[tok] += d_enc[0, t, :] * 0.5

        # === UPDATE EMBEDDINGS (use optimizer) ===
        self._opt_update(self.embedding, d_embed, 'embedding')
    
    def generate(self, input_text, tokenizer, max_length=50, temperature=1.0):
        self.training = False   # disable dropout during inference
        input_tokens = tokenizer.encode(input_text)
        input_seq = self.xp.array(input_tokens).reshape(1, -1)
        
        encoder_output = self.encode(input_seq)
        
        generated = [tokenizer.word2idx['<START>']]
        
        for _ in range(max_length):
            # Cap to max_seq_len so positional encoding never overflows
            window = generated[-self.max_seq_len:]
            target_seq = self.xp.array(window).reshape(1, -1)
            decoder_output = self.decode(target_seq, encoder_output)
            
            logits = self.xp.dot(decoder_output[:, -1:, :], self.output_weights) + self.output_bias
            logits = logits.reshape(-1)
            
            if self.gpu_available:
                logits = cp.asnumpy(logits)
            
            probs = np.exp(logits / temperature)
            probs = probs / np.sum(probs)
            
            next_token = np.random.choice(len(probs), p=probs)
            
            # Don't allow <END> too early (min 2 content tokens after <START>)
            if next_token == tokenizer.word2idx['<END>'] and len(generated) > 2:
                break
            
            if next_token != tokenizer.word2idx['<END>']:
                generated.append(next_token)
        
        self.training = True    # restore training mode
        return tokenizer.decode(generated[1:])
    
    def save(self, filepath):
        model_data = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'max_seq_len': self.max_seq_len,
            'learning_rate': self.learning_rate,
            'embedding': cp.asnumpy(self.embedding) if self.gpu_available else self.embedding,
            'output_weights': cp.asnumpy(self.output_weights) if self.gpu_available else self.output_weights,
            'output_bias': cp.asnumpy(self.output_bias) if self.gpu_available else self.output_bias,
            'encoder_layers': [],
            'decoder_layers': []
        }
        
        for layer in self.encoder_layers:
            layer_data = {}
            for key, param in layer.items():
                layer_data[key] = cp.asnumpy(param) if self.gpu_available else param
            model_data['encoder_layers'].append(layer_data)
        
        for layer in self.decoder_layers:
            layer_data = {}
            for key, param in layer.items():
                layer_data[key] = cp.asnumpy(param) if self.gpu_available else param
            model_data['decoder_layers'].append(layer_data)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Transformer model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = TransformerChatbot(
            vocab_size=model_data['vocab_size'],
            embed_dim=model_data['embed_dim'],
            num_heads=model_data['num_heads'],
            num_layers=model_data['num_layers'],
            ff_dim=model_data['ff_dim'],
            max_seq_len=model_data['max_seq_len'],
            learning_rate=model_data['learning_rate']
        )
        
        xp = model.xp
        model.embedding = xp.array(model_data['embedding'])
        model.output_weights = xp.array(model_data['output_weights'])
        model.output_bias = xp.array(model_data['output_bias'])
        
        for i, layer_data in enumerate(model_data['encoder_layers']):
            for key, param in layer_data.items():
                model.encoder_layers[i][key] = xp.array(param)
        
        for i, layer_data in enumerate(model_data['decoder_layers']):
            for key, param in layer_data.items():
                model.decoder_layers[i][key] = xp.array(param)
        
        print(f"Transformer model loaded from {filepath}")
        return model
