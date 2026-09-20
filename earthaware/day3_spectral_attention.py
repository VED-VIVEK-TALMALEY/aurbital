

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralAttentionModule(nn.Module):

    def __init__(self, 
                 num_bands=13, 
                 embed_dim=768,
                 num_heads=8):
        
        super().__init__()
        
        self.num_bands = num_bands
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.band_embeddings = nn.Parameter(
            torch.randn(num_bands, embed_dim)
        )

        self.query_proj = nn.Linear(embed_dim, embed_dim)

        self.key_proj = nn.Linear(embed_dim, embed_dim)

        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

        self.scale = self.head_dim ** -0.5

        nn.init.trunc_normal_(self.band_embeddings, std=0.02)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x, return_attention_weights=False):
        
        batch_size, num_patches, embed_dim = x.shape

        Q = self.query_proj(x)  

        band_keys = self.key_proj(self.band_embeddings)  
        band_keys = band_keys.unsqueeze(0).expand(batch_size, -1, -1)

        band_values = self.value_proj(self.band_embeddings)
        band_values = band_values.unsqueeze(0).expand(batch_size, -1, -1)

        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  
        
        K = band_keys.view(batch_size, self.num_bands, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  
        
        V = band_values.view(batch_size, self.num_bands, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)  

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, num_patches, embed_dim)

        attn_output = self.out_proj(attn_output)

        output = x + attn_output
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output

class SimplifiedSpectralAttention(nn.Module):

    def __init__(self, num_bands=13, embed_dim=768):
        super().__init__()
        
        self.num_bands = num_bands
        self.embed_dim = embed_dim

        self.band_embeddings = nn.Parameter(
            torch.randn(num_bands, embed_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_bands),
            nn.Softmax(dim=-1)
        )

        self.value_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, return_attention_weights=False):
        
        batch_size, num_patches, embed_dim = x.shape

        attn_weights = self.attention(x)  

        band_values = self.value_proj(self.band_embeddings)

        attended = torch.matmul(attn_weights, band_values)

        output = x + attended
        
        if return_attention_weights:
            return output, attn_weights
        else:
            return output

if __name__ == "__main__":

    print("="*60)
    print("TESTING SPECTRAL ATTENTION MODULE")
    print("="*60)

    batch_size = 2
    num_patches = 64  
    embed_dim = 768
    
    dummy_input = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"\nInput: {dummy_input.shape}")
    print("  (batch=2, patches=64, embed_dim=768)")

    print("\n1. Multi-Head Spectral Attention:")
    attn_module = SpectralAttentionModule(
        num_bands=13,
        embed_dim=768,
        num_heads=8
    )
    
    output, weights = attn_module(dummy_input, return_attention_weights=True)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"     (batch, heads, patches, bands)")

    avg_attention = weights.mean(dim=(0, 1))  
    
    print(f"\n   Average attention per band:")
    band_importance = avg_attention.mean(dim=0)  
    
    band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                  'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    
    for i, (name, importance) in enumerate(zip(band_names, band_importance)):
        print(f"     {name}: {importance:.4f}")

    print("\n2. Simplified Spectral Attention:")
    simple_attn = SimplifiedSpectralAttention(
        num_bands=13,
        embed_dim=768
    )
    
    output2, weights2 = simple_attn(dummy_input, return_attention_weights=True)
    
    print(f"   Output shape: {output2.shape}")
    print(f"   Attention weights: {weights2.shape}")

    params1 = sum(p.numel() for p in attn_module.parameters())
    params2 = sum(p.numel() for p in simple_attn.parameters())
    
    print(f"\n3. Parameter Count:")
    print(f"   Multi-head: {params1:,}")
    print(f"   Simplified: {params2:,}")
    print(f"   Ratio: {params1/params2:.1f}x")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("- Use multi-head for best performance")
    print("- Use simplified if memory is tight")
    print("="*60)
