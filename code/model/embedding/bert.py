import torch.nn as nn

def token_emb_layer(weights_matrix, non_trainable=True):
        vocab_size, embed_size = weights_matrix.size()
        token_emb_layer = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        token_emb_layer.load_state_dict({'weight': weights_matrix})
        if non_trainable:
            token_emb_layer.weight.requires_grad = False

        return token_emb_layer

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, weights_matrix, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = token_emb_layer(weights_matrix)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence)
        return self.dropout(x)