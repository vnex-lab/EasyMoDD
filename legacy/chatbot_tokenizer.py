import re
import json
import pickle
from typing import List, Dict, Tuple
from collections import Counter

class ChatbotTokenizer:
    """
    Simple tokenizer for chatbot text processing
    """
    
    def __init__(self, max_vocab_size: int = 10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.vocab_size = 4
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words with improved regex for code and symbols
        """
        # Convert to lowercase
        text = text.lower()
        # Keep basic punctuation and common code symbols as separate tokens
        text = re.sub(r'([.,!?(){}\[\]:;=+\-*/<>_])', r' \1 ', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        tokens = text.strip().split()
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Get most common tokens
        most_common = token_counts.most_common(self.max_vocab_size - 4)  # -4 for special tokens
        
        # Add to vocabulary
        for token, _ in most_common:
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token
        
        self.vocab_size = len(self.word2idx)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices
        
        Args:
            text: Input text
            add_special_tokens: Whether to add START and END tokens
        
        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx['<START>'])
        
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx['<UNK>'])
        
        if add_special_tokens:
            indices.append(self.word2idx['<END>'])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text string
        """
        tokens = []
        special_tokens = {'<PAD>', '<START>', '<END>', '<UNK>'}
        
        for idx in indices:
            if idx in self.idx2word:
                token = self.idx2word[idx]
                if skip_special_tokens and token in special_tokens:
                    continue
                tokens.append(token)
        
        # Join tokens and clean up punctuation spacing
        text = ' '.join(tokens)
        # Remove spaces before common punctuation
        text = re.sub(r'\s+([.,!?;:\)\]\}])', r'\1', text)
        # Remove spaces after opening brackets
        text = re.sub(r'([\(\[\{])\s+', r'\1', text)
        # Fix contractions like i ' m -> i'm
        text = re.sub(r"(\w)\s+'\s+(\w)", r"\1'\2", text)
        return text
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'max_vocab_size': self.max_vocab_size,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.max_vocab_size = data['max_vocab_size']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.vocab_size = data['vocab_size']
    
    def get_vocab_info(self) -> Dict:
        """Get vocabulary information"""
        return {
            'vocab_size': self.vocab_size,
            'max_vocab_size': self.max_vocab_size,
            'special_tokens': ['<PAD>', '<START>', '<END>', '<UNK>']
        }
