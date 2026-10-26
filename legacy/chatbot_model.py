import json
import pickle
from typing import List, Dict, Tuple, Optional

# GPU Support - Try CuPy first, fallback to NumPy
try:
    import cupy as cp
    # Test if GPU is actually usable with all features we need
    try:
        # Test basic array operations
        test_arr = cp.array([1, 2, 3])
        # Test random number generation (this is where DLL errors often happen)
        cp.random.seed(42)
        test_random = cp.random.randn(5, 5)
        # Test math operations
        _ = cp.exp(test_random)
        
        # If all tests pass, use GPU
        np = cp
        GPU_AVAILABLE = True
        DEVICE = "GPU"
        print("GPU detected and verified - using CuPy for acceleration!")
    except Exception as e:
        # CuPy installed but GPU not usable (driver issues, DLL errors, etc.)
        print(f"GPU detected but not usable ({str(e)[:50]}...) - falling back to CPU")
        import numpy as np
        GPU_AVAILABLE = False
        DEVICE = "CPU"
except ImportError:
    # CuPy not installed
    print("ℹ️ CuPy not installed - using CPU mode (install cupy-cuda12x for GPU support)")
    import numpy as np
    GPU_AVAILABLE = False
    DEVICE = "CPU"

class VnexAIChatbot:
    """
    VnexAI Chatbot - A simple sequence-to-sequence chatbot built from scratch
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 max_length: int = 50, learning_rate: float = 0.01):
        """
        Initialize the VnexAI Chatbot
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
            max_length: Maximum sequence length
            learning_rate: Learning rate for training
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.device = DEVICE
        self.gpu_available = GPU_AVAILABLE
        
        # Initialize weights
        self._initialize_weights()
        
        # Training history
        self.training_history = {
            'loss': [],
            'perplexity': []
        }
    
    def get_device_info(self) -> str:
        """Get information about the device being used"""
        if self.gpu_available:
            return f"🚀 GPU Acceleration ACTIVE ({np.__name__})"
        else:
            return f"💻 CPU Mode ({np.__name__})"
    
    def _initialize_weights(self):
        """Initialize all model weights using Xavier/He initialization for better gradient flow"""
        global np, GPU_AVAILABLE, DEVICE
        
        try:
            np.random.seed(42)
            
            # Xavier/He initialization scales
            emb_scale = 0.1
            rnn_scale = np.sqrt(2.0 / (self.embedding_dim + self.hidden_dim))
            out_scale = np.sqrt(2.0 / (self.hidden_dim + self.vocab_size))
            
            # Embedding layer
            self.embedding = np.random.randn(self.vocab_size, self.embedding_dim) * emb_scale
            
            # Encoder RNN weights
            self.Wxh_enc = np.random.randn(self.embedding_dim, self.hidden_dim) * rnn_scale
            self.Whh_enc = np.random.randn(self.hidden_dim, self.hidden_dim) * rnn_scale
            self.bh_enc = np.zeros((1, self.hidden_dim))
            
            # Decoder RNN weights
            self.Wxh_dec = np.random.randn(self.embedding_dim, self.hidden_dim) * rnn_scale
            self.Whh_dec = np.random.randn(self.hidden_dim, self.hidden_dim) * rnn_scale
            self.bh_dec = np.zeros((1, self.hidden_dim))
            
            # Output layer
            self.Why = np.random.randn(self.hidden_dim, self.vocab_size) * out_scale
            self.by = np.zeros((1, self.vocab_size))
            
        except Exception as e:
            # GPU failed during initialization, fall back to CPU
            if GPU_AVAILABLE:
                import numpy as numpy_fallback
                np = numpy_fallback
                GPU_AVAILABLE = False
                DEVICE = "CPU"
                self.gpu_available = False
                self.device = "CPU"
                
                # Retry initialization with NumPy
                np.random.seed(42)
                self.embedding = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
                self.Wxh_enc = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
                self.Whh_enc = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
                self.bh_enc = np.zeros((1, self.hidden_dim))
                self.Wxh_dec = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.01
                self.Whh_dec = np.random.randn(self.hidden_dim, self.hidden_dim) * 0.01
                self.bh_dec = np.zeros((1, self.hidden_dim))
                self.Why = np.random.randn(self.hidden_dim, self.vocab_size) * 0.01
                self.by = np.zeros((1, self.vocab_size))
            else:
                raise e
    
    def _sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def _tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)
    
    def _softmax(self, x):
        """Softmax activation with numerical stability"""
        # Subtract max for stability
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        # Add epsilon to denominator to prevent division by zero
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True) + 1e-10
        return exp_x / sum_exp
    
    def encode(self, input_seq: np.ndarray) -> np.ndarray:
        """
        Encode input sequence
        
        Args:
            input_seq: Input token indices [seq_len]
        
        Returns:
            Final hidden state [hidden_dim]
        """
        h = np.zeros((1, self.hidden_dim))
        
        for t in range(len(input_seq)):
            if input_seq[t] < self.vocab_size:
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_enc) + np.dot(h, self.Whh_enc) + self.bh_enc)
        
        return h
    
    def decode(self, hidden_state: np.ndarray, target_seq: Optional[np.ndarray] = None, 
               generate: bool = False, temperature: float = 0.8) -> Tuple[np.ndarray, List[np.ndarray], Optional[List[int]]]:
        """
        Decode from hidden state
        
        Args:
            hidden_state: Initial hidden state from encoder
            target_seq: Target sequence for training (optional)
            generate: Whether to generate or use teacher forcing
            temperature: Sampling temperature (higher = more random, lower = more focused)
        
        Returns:
            outputs: Output probabilities for each timestep
            hidden_states: Hidden states for each timestep
            generated_tokens: List of generated token indices (only in generate mode)
        """
        h = hidden_state
        outputs = []
        hidden_states = [h]
        generated_tokens = []
        
        # Start token (0) or first target token
        if target_seq is not None and not generate:
            seq_len = len(target_seq)
            for t in range(seq_len):
                x_t = self.embedding[target_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_dec) + np.dot(h, self.Whh_dec) + self.bh_dec)
                y = np.dot(h, self.Why) + self.by
                output = self._softmax(y)
                outputs.append(output)
                hidden_states.append(h)
        else:
            # Generation mode with temperature sampling
            x_t = self.embedding[1].reshape(1, -1)  # Start with <START> token (index 1)
            for t in range(self.max_length):
                h = self._tanh(np.dot(x_t, self.Wxh_dec) + np.dot(h, self.Whh_dec) + self.bh_dec)
                y = np.dot(h, self.Why) + self.by
                output = self._softmax(y / temperature)  # Apply temperature
                outputs.append(output)
                hidden_states.append(h)
                
                # Sample next token with temperature
                # Flatten output and convert to probabilities
                probs = output.flatten()
                
                # Convert to CPU for sampling if using GPU
                if GPU_AVAILABLE and hasattr(probs, 'get'):
                    probs = probs.get()
                
                # Sample from probability distribution instead of argmax
                probs = probs / probs.sum()  # Normalize
                # Use size=1 for CuPy compatibility
                import numpy as real_np
                next_token = int(real_np.random.choice(len(probs), size=1, p=probs)[0])
                
                generated_tokens.append(next_token)
                
                if next_token == 2:  # <END> token is index 2 (not 1 which is <START>)
                    break
                x_t = self.embedding[next_token].reshape(1, -1)
        
        return np.array(outputs), hidden_states, generated_tokens if generate else None
    
    def forward(self, input_seq: np.ndarray, target_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Forward pass
        
        Args:
            input_seq: Input sequence [seq_len]
            target_seq: Target sequence [seq_len]
        
        Returns:
            outputs: Predicted probabilities
            encoder_hidden: Encoder hidden state
            decoder_hidden_states: Decoder hidden states
        """
        encoder_hidden = self.encode(input_seq)
        outputs, decoder_hidden_states, _ = self.decode(encoder_hidden, target_seq, generate=False)
        return outputs, encoder_hidden, decoder_hidden_states
    
    def compute_loss(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        outputs[t] is the prediction made AFTER seeing targets[t] as input,
        so the correct label is targets[t+1] (next-token prediction).
        """
        loss = 0
        count = 0
        for t in range(len(outputs)):
            if t + 1 < len(targets):
                loss += -np.log(outputs[t, 0, targets[t + 1]] + 1e-10)
                count += 1
        avg_loss = loss / max(count, 1)
        if GPU_AVAILABLE and hasattr(avg_loss, 'get'):
            return float(avg_loss.get())
        return float(avg_loss)
    
    def backward(self, input_seq: np.ndarray, target_seq: np.ndarray, 
                outputs: np.ndarray, encoder_hidden: np.ndarray, 
                decoder_hidden_states: List) -> Dict[str, np.ndarray]:
        """
        Backward pass using simplified gradient computation
        
        Returns:
            gradients: Dictionary of gradients
        """
        # Initialize gradients
        dWxh_enc = np.zeros_like(self.Wxh_enc)
        dWhh_enc = np.zeros_like(self.Whh_enc)
        dbh_enc = np.zeros_like(self.bh_enc)
        
        dWxh_dec = np.zeros_like(self.Wxh_dec)
        dWhh_dec = np.zeros_like(self.Whh_dec)
        dbh_dec = np.zeros_like(self.bh_dec)
        
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dembedding = np.zeros_like(self.embedding)
        
        # Backward through decoder
        # decoder_hidden_states[0]   = encoder hidden (initial)
        # decoder_hidden_states[t+1] = hidden state produced at step t
        # output[t] was computed from hidden_states[t+1], not [t]
        dh_next = np.zeros((1, self.hidden_dim))
        
        for t in reversed(range(len(target_seq) - 1)):
            if t < len(outputs) and t + 1 < len(target_seq):
                # FIXED: predict targets[t+1] from step t (next-token prediction)
                dy = outputs[t].copy()
                dy[0, target_seq[t + 1]] -= 1

                # FIXED: output[t] = Why @ h_{t+1}, so gradient uses h_{t+1}
                dWhy += np.dot(decoder_hidden_states[t + 1].T, dy)
                dby += dy

                dh = np.dot(dy, self.Why.T) + dh_next
                # FIXED: tanh derivative uses h_{t+1} (the state that was output at step t)
                dh_raw = dh * (1 - decoder_hidden_states[t + 1] ** 2)

                # Decoder RNN gradients
                x_t = self.embedding[target_seq[t]].reshape(1, -1)
                dWxh_dec += np.dot(x_t.T, dh_raw)
                # FIXED: recurrent gradient uses h_t = decoder_hidden_states[t]
                dWhh_dec += np.dot(decoder_hidden_states[t].T, dh_raw)
                dbh_dec += dh_raw

                # Embedding gradient (input token at step t)
                dembedding[target_seq[t]] += np.dot(dh_raw, self.Wxh_dec.T).flatten()

                dh_next = np.dot(dh_raw, self.Whh_dec.T)
        
        # Simplified encoder gradient (only propagate from final state)
        dh_enc = dh_next
        encoder_states = []
        h = np.zeros((1, self.hidden_dim))
        for t in range(len(input_seq)):
            if input_seq[t] < self.vocab_size:
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                h = self._tanh(np.dot(x_t, self.Wxh_enc) + np.dot(h, self.Whh_enc) + self.bh_enc)
                encoder_states.append(h)
        
        for t in reversed(range(len(input_seq))):
            if input_seq[t] < self.vocab_size and t < len(encoder_states):
                dh_raw = dh_enc * (1 - encoder_states[t] ** 2)
                x_t = self.embedding[input_seq[t]].reshape(1, -1)
                dWxh_enc += np.dot(x_t.T, dh_raw)
                if t > 0:
                    dWhh_enc += np.dot(encoder_states[t-1].T, dh_raw)
                dbh_enc += dh_raw
                
                dembedding[input_seq[t]] += np.dot(dh_raw, self.Wxh_enc.T).flatten()
                dh_enc = np.dot(dh_raw, self.Whh_enc.T)
        
        # Clip ALL gradients including biases to prevent explosions
        for grad in [dWxh_enc, dWhh_enc, dbh_enc,
                     dWxh_dec, dWhh_dec, dbh_dec,
                     dWhy, dby, dembedding]:
            np.clip(grad, -5, 5, out=grad)
        
        return {
            'dWxh_enc': dWxh_enc,
            'dWhh_enc': dWhh_enc,
            'dbh_enc': dbh_enc,
            'dWxh_dec': dWxh_dec,
            'dWhh_dec': dWhh_dec,
            'dbh_dec': dbh_dec,
            'dWhy': dWhy,
            'dby': dby,
            'dembedding': dembedding
        }
    
    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update weights using gradients"""
        self.Wxh_enc -= self.learning_rate * gradients['dWxh_enc']
        self.Whh_enc -= self.learning_rate * gradients['dWhh_enc']
        self.bh_enc -= self.learning_rate * gradients['dbh_enc']
        
        self.Wxh_dec -= self.learning_rate * gradients['dWxh_dec']
        self.Whh_dec -= self.learning_rate * gradients['dWhh_dec']
        self.bh_dec -= self.learning_rate * gradients['dbh_dec']
        
        self.Why -= self.learning_rate * gradients['dWhy']
        self.by -= self.learning_rate * gradients['dby']
        
        self.embedding -= self.learning_rate * gradients['dembedding']
    
    def step_lr(self, decay: float = 0.99):
        """Call once per epoch to decay the learning rate by 1% (not 2% — keeps
        learning signal alive much longer into training)."""
        if self.learning_rate > 1e-5:
            self.learning_rate *= decay

    def train_step(self, input_seq: np.ndarray, target_seq: np.ndarray) -> float:
        """Single training step"""
        outputs, encoder_hidden, decoder_hidden_states = self.forward(input_seq, target_seq)
        loss = self.compute_loss(outputs, target_seq)
        gradients = self.backward(input_seq, target_seq, outputs, encoder_hidden, decoder_hidden_states)
        self.update_weights(gradients)
        return loss
    
    def train_batch(self, input_seqs: list, target_seqs: list) -> float:
        """Batch training step - processes multiple samples at once for GPU efficiency"""
        batch_size = len(input_seqs)
        total_loss = 0.0
        
        # Initialize accumulated gradients
        accumulated_grads = {
            'dWxh_enc': np.zeros_like(self.Wxh_enc),
            'dWhh_enc': np.zeros_like(self.Whh_enc),
            'dbh_enc': np.zeros_like(self.bh_enc),
            'dWxh_dec': np.zeros_like(self.Wxh_dec),
            'dWhh_dec': np.zeros_like(self.Whh_dec),
            'dbh_dec': np.zeros_like(self.bh_dec),
            'dWhy': np.zeros_like(self.Why),
            'dby': np.zeros_like(self.by),
            'dembedding': np.zeros_like(self.embedding)
        }
        
        # Process each sample in the batch
        for input_seq, target_seq in zip(input_seqs, target_seqs):
            # Forward pass
            outputs, encoder_hidden, decoder_hidden_states = self.forward(input_seq, target_seq)
            
            # Compute loss
            loss = self.compute_loss(outputs, target_seq)
            total_loss += loss
            
            # Backward pass
            gradients = self.backward(input_seq, target_seq, outputs, encoder_hidden, decoder_hidden_states)
            
            # Accumulate gradients
            for key in accumulated_grads:
                accumulated_grads[key] += gradients[key]
        
        # Average gradients across batch
        for key in accumulated_grads:
            accumulated_grads[key] /= batch_size
        
        # Update weights with averaged gradients
        self.update_weights(accumulated_grads)
        
        # Return average loss as Python float (works for both NumPy and CuPy)
        avg_loss = total_loss / batch_size
        if GPU_AVAILABLE and hasattr(avg_loss, 'get'):
            return float(avg_loss.get())  # CuPy to Python float
        return float(avg_loss)  # NumPy to Python float
    
    def generate_response(self, input_seq: np.ndarray, temperature: float = 0.8) -> np.ndarray:
        """
        Generate a response for given input
        
        Args:
            input_seq: Input token sequence
            temperature: Sampling temperature (0.1-2.0, default 0.8)
                        Lower = more focused, Higher = more creative
        
        Returns:
            Generated token sequence
        """
        encoder_hidden = self.encode(input_seq)
        outputs, _, generated_tokens = self.decode(encoder_hidden, generate=True, temperature=temperature)
        
        # Return the generated tokens (already sampled with temperature)
        # Remove the final <END> token if present
        if generated_tokens and generated_tokens[-1] == 1:
            return np.array(generated_tokens[:-1])
        return np.array(generated_tokens) if generated_tokens else np.array([])
    
    def _to_cpu(self, array):
        """Convert array to CPU (NumPy) for saving"""
        if GPU_AVAILABLE and hasattr(array, 'get'):
            return array.get()  # CuPy to NumPy
        return array
    
    def _to_device(self, array):
        """Convert array to current device"""
        if GPU_AVAILABLE:
            import cupy as cp
            if not isinstance(array, cp.ndarray):
                return cp.array(array)
        return array
    
    def save_model(self, filepath: str):
        """Save model to binary file"""
        model_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'embedding': self._to_cpu(self.embedding),
            'Wxh_enc': self._to_cpu(self.Wxh_enc),
            'Whh_enc': self._to_cpu(self.Whh_enc),
            'bh_enc': self._to_cpu(self.bh_enc),
            'Wxh_dec': self._to_cpu(self.Wxh_dec),
            'Whh_dec': self._to_cpu(self.Whh_dec),
            'bh_dec': self._to_cpu(self.bh_dec),
            'Why': self._to_cpu(self.Why),
            'by': self._to_cpu(self.by),
            'training_history': self.training_history
        }
        
        # Save as binary file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model from binary file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_dim = model_data['hidden_dim']
        self.max_length = model_data['max_length']
        self.learning_rate = model_data['learning_rate']
        
        # Convert to appropriate device (GPU if available)
        self.embedding = self._to_device(model_data['embedding'])
        self.Wxh_enc = self._to_device(model_data['Wxh_enc'])
        self.Whh_enc = self._to_device(model_data['Whh_enc'])
        self.bh_enc = self._to_device(model_data['bh_enc'])
        self.Wxh_dec = self._to_device(model_data['Wxh_dec'])
        self.Whh_dec = self._to_device(model_data['Whh_dec'])
        self.bh_dec = self._to_device(model_data['bh_dec'])
        self.Why = self._to_device(model_data['Why'])
        self.by = self._to_device(model_data['by'])
        self.training_history = model_data['training_history']
