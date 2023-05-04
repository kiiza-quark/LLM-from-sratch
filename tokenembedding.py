import torch

class TokenEmbedding(torch.nn.Module):
    """
    Pytorch module that converts tokens into embeddings.
    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """
    def __init__(
        self,
        embedding_dimension,
        number_of_tokens
    ):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(
        num_embeddings=number_of_tokens,
        embedding_dim=embedding_dimension
        )

    def forward(self, x):
        return self.embedding_layer(x)
