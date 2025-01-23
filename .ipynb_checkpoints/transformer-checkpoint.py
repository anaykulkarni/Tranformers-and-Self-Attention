import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def positional_encoding(max_len, embed_dim):
    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # self.token_embedding = nn.Embedding(max_len, embed_dim) # Learnable encoding
        self.position_embedding = positional_encoding(max_len, embed_dim)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Compute token and position embeddings
        seq_len = x.size(1)
        
        # positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        # x = self.token_embedding(x) + self.position_embedding(positions)

        token_embedding = self.token_embedding(x)
        position_encoding = self.position_embedding[:seq_len, :].unsqueeze(0) # Shape: (1, seq_len, embed_dim)

        x = token_embedding + position_encoding

        x = self.dropout(x)

        attention_maps = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attn_map = layer(x)
            attention_maps.append(attn_map)
        
        return x, attention_maps

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, alibi=False, scaling_factor=1.0):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, alibi, scaling_factor)
        
        # Feedforward layers
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attn_output, attn_map = self.multi_head_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection and layer normalization
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_map

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, alibi=False, scaling_factor=1.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.alibi = alibi
        self.scaling_factor = scaling_factor
        
        # Linear layers for query, key, value, and final output
        self.query = nn.Linear(embed_dim, embed_dim) # This linear transformation keeps the shape of the output same as input
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def generate_alibi_encoding(self, seq_length, num_heads):
        # Create a bias matrix where each entry is proportional to the negative distance between tokens
        # A matrix of shape (seq_length, seq_length) such that mat(i, j) = i-j
        alibi_encoding = torch.arange(seq_length).view(1, -1) - torch.arange(seq_length).view(-1, 1)
        # Make both upper and lower triangular portions negative
        alibi_encoding = -self.scaling_factor*torch.abs(alibi_encoding.float())
        # generate 'num_heads' such matrices; (num_heads, seq_length, seq_length)
        alibi_encoding = alibi_encoding.unsqueeze(0).repeat(num_heads, 1, 1)
        return alibi_encoding
    
    def scaled_dot_product_attention(self, Q, K, V, alibi_encoding=None):
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # Each batch and attention head is processed independently (parallely) So we ignore first two dimensions
        # attention_weights: (seq_len, head_dim) * (head_dim, seq_len) = (seq_len, seq_len)
        # For each (batch_size, num_head) pair we get a (seq_len, seq_len) matrix of attention scores.
        if self.alibi and alibi_encoding is not None:
            scores += alibi_encoding
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        # Linear projections
        Q = self.query(x) # Q: (batch_size, seq_len, embed_dim)
        K = self.key(x)   # K: (batch_size, seq_len, embed_dim)
        V = self.value(x) # V: (batch_size, seq_len, embed_dim)
        
        # Split each Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        # Makes Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 

        # If using alibi encoding
        if self.alibi:
            alibi_encoding = self.generate_alibi_encoding(seq_length, self.num_heads)
            attn_output, attention_map = self.scaled_dot_product_attention(Q, K, V, alibi_encoding)
        else:  
            # Apply scaled dot-product attention to each head
            attn_output, attention_map = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(attn_output)
        
        return output, attention_map

class FeedforwardClassifier(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_classes, dropout=0.1):
        super(FeedforwardClassifier, self).__init__()
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)  # First linear layer
        self.relu = nn.ReLU()                    # Activation function
        self.fc2 = nn.Linear(ff_dim, num_classes)  # Second linear layer
        self.dropout = nn.Dropout(dropout)       # Dropout layer for regularization
    
    def forward(self, x):
        x = self.fc1(x)                          # Pass through first linear layer
        x = self.relu(x)                         # Apply ReLU activation
        x = self.dropout(x)                      # Apply dropout
        x = self.fc2(x)                          # Pass through second linear layer
        return x                                  # Return logits for each class

class TransformerModelwithCLS(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len, dropout=0.1):
        super(TransformerModelwithCLS, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout)
        self.classifier = FeedforwardClassifier(embed_dim, ff_dim, num_classes, dropout)
    
    def forward(self, x):
        encoder_output, attention_maps = self.encoder(x)  # Get output and attention maps from encoder
        # Average the output embeddings across the sequence length
        encoder_output = encoder_output.mean(dim=1)  # Shape: (batch_size, embed_dim)
        logits = self.classifier(encoder_output)      # Get predictions from classifier
        return logits, attention_maps  # Return logits and attention maps


##---------------------------------------------------------------------------------------------------------
## PART 2
##---------------------------------------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Encoding
        # self.token_embedding = nn.Embedding(max_len, embed_dim) # Learnable encoding
        self.position_embedding = positional_encoding(max_len, embed_dim)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])

        # Normalization layer before output
        self.layerNorm = nn.LayerNorm(embed_dim)
        
        # Output layer to convert decoder output to vocab probabilities
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get sequence length
        seq_len = x.size(1)

        # positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        # x = self.token_embedding(x) + self.position_embedding(positions)
        token_embedding = self.token_embedding(x)
        
        # Shape: (1, seq_len, embed_dim) after adding batch dimension
        position_encoding = self.position_embedding[:seq_len, :].unsqueeze(0) 

        x = token_embedding + position_encoding

        x = self.dropout(x)

        attention_maps = []
        
        # Pass through each decoder layer
        for layer in self.layers:
            x, attn_map = layer(x)
            attention_maps.append(attn_map)
        
        x = self.layerNorm(x)

        # Project to vocabulary size to get logits
        logits = self.output_layer(x)
        
        return logits, attention_maps

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(TransformerDecoderLayer, self).__init__()
        
        # Masked self-attention
        self.masked_attention = MaskedMultiHeadAttention(embed_dim, num_heads)
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Masked self-attention with residual connection and layer norm
        attn_output, attn_map = self.masked_attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_map

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by num heads."
        
        # Linear layers for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output linear layer
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        # Linear projections
        Q = self.query(x) # Q: (batch_size, seq_len, embed_dim)
        K = self.key(x)   # K: (batch_size, seq_len, embed_dim)
        V = self.value(x) # V: (batch_size, seq_len, embed_dim)
        
        # Split each Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        # Makes Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute attention probabilities
        attention_weights = F.softmax(attn_scores, dim=-1)
        # Compute the attention output by multiplying probabilities with values
        attn_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out(attn_output)
        
        return output, attention_weights

##---------------------------------------------------------------------------------------------------------
## PART 3
##---------------------------------------------------------------------------------------------------------
class TransformerEncoderWithAliBi(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout=0.1, alibi=True, scaling_factor=1.0):
        super(TransformerEncoderWithAliBi, self).__init__()
        
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout, alibi, scaling_factor)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Compute token and position embeddings
        seq_len = x.size(1)

        x = self.token_embedding(x)

        x = self.dropout(x)

        attention_maps = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attn_map = layer(x)
            attention_maps.append(attn_map)
        
        return x, attention_maps

class TransformerModelwithCLS_AliBi(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, num_classes, max_len, dropout=0.1, alibi=True, scaling_factor=1.0):
        super(TransformerModelwithCLS_AliBi, self).__init__()
        self.encoder = TransformerEncoderWithAliBi(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout, alibi, scaling_factor)
        self.classifier = FeedforwardClassifier(embed_dim, ff_dim, num_classes, dropout)
    
    def forward(self, x):
        encoder_output, attention_maps = self.encoder(x)  # Get output and attention maps from encoder
        # Average the output embeddings across the sequence length
        encoder_output = encoder_output.mean(dim=1)  # Shape: (batch_size, embed_dim)
        logits = self.classifier(encoder_output)      # Get predictions from classifier
        return logits, attention_maps  # Return logits and attention maps