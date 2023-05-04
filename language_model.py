from decoder import DecoderStack
from tokenembedding import TokenEmbedding
from positionencoding import  PositionalEncoding

import torch

class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """
    def __init__(
    self,
    number_of_tokens, # The number of tokens in the vocabulary
    max_sequence_length=512, # The maximum sequence length to use for attention
    embedding_dimension=512, # The dimension of the token embeddings
    number_of_layers=6, # The number of decoder layers to use
    number_of_heads=4, # The number of attention heads to use
    feed_forward_dimension=None, # The dimension of the feed forward layer
    dropout_rate=0.1 # The dropout rate to use
    ):
        
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if feed_forward_dimension is None:
        # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension
        
        self.dropout_rate = dropout_rate
        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)
        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension,
        max_sequence_length)
        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)
        # Create the decoder stack
        self.decoder = DecoderStack(
        embedding_dimension=embedding_dimension,
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        feed_forward_dimension=self.feed_forward_dimension,
        dropout_rate=dropout_rate,
        max_sequence_length=max_sequence_length
        )# Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)
        
    def forward(self, x, mask):
    # Compute the token embeddings
    # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)
    # Compute the positional encoding
    # positional_encoding dimensions are: (batch_size, sequence_length,embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)
    # Post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)
        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)
        return lm_head_outputs
    
class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the
    vocabulary size.
    """
    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)
    
    def forward(self, x):
        """
        Compute the language model head.
        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """# Compute the linear layer
     # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)
        return linear_output
    

class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.
    """
    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = self.model.max_sequence_length
    
    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]
        output = self.model(inp, mask)
        return output, target
    
    def next_token_probabilities(self, x, mask, temperature=1.0):
        """
        Calculate the token probabilities for the next token in the sequence.
        """
        logits = self.model(x, mask)[:, -1]
        # Apply the temperature
        if temperature != 1.0:
            logits = logits / temperature
        # Apply the softmax
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities