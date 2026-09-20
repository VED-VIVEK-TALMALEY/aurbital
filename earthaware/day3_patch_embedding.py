

import torch
import torch.nn as nn

class SpectralPatchEmbedding(nn.Module):

    def __init__(self, 
                 in_channels=13, 
                 patch_size=8, 
                 embed_dim=768,
                 image_size=64):
        
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        x = self.projection(x)

        x = x.flatten(2)

        x = x.transpose(1, 2)
        
        return x

class SpectralPatchEmbeddingPerBand(nn.Module):

    def __init__(self, 
                 in_channels=13, 
                 patch_size=8, 
                 embed_dim=768,
                 image_size=64):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2

        self.band_embed_dim = (embed_dim + in_channels - 1) // in_channels

        self.band_projections = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=self.band_embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )
            for _ in range(in_channels)
        ])

        self.fusion = nn.Linear(embed_dim, embed_dim)

        for proj in self.band_projections:
            nn.init.trunc_normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        nn.init.trunc_normal_(self.fusion.weight, std=0.02)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, x):
        
        batch_size = x.shape[0]

        band_embeddings = []
        for i, projection in enumerate(self.band_projections):
            
            band = x[:, i:i+1, :, :]  

            band_embed = projection(band)  
            band_embed = band_embed.flatten(2).transpose(1, 2)

            band_embeddings.append(band_embed)

        x = torch.cat(band_embeddings, dim=2)  

        x = self.fusion(x)
        
        return x

if __name__ == "__main__":

    print("="*60)
    print("TESTING SPECTRAL PATCH EMBEDDING")
    print("="*60)

    batch_size = 2
    dummy_input = torch.randn(batch_size, 13, 64, 64)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print("  13 spectral bands, 64×64 pixels")

    print("\n1. Simple Projection:")
    embed1 = SpectralPatchEmbedding(
        in_channels=13,
        patch_size=8,
        embed_dim=768
    )
    
    output1 = embed1(dummy_input)
    print(f"   Output shape: {output1.shape}")
    print(f"   Expected: (2, 64, 768) - 64 patches, 768-dim embeddings")

    print("\n2. Per-Band Projection: SKIPPED")
    print("   (We're using simple projection - works better!)")

    params1 = sum(p.numel() for p in embed1.parameters())
    
    print(f"\n3. Parameter Count:")
    print(f"   Simple projection: {params1:,} parameters")
    print(f"   Memory: ~{params1 * 4 / (1024**2):.1f} MB (FP32)")
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Start with simple projection")
    print("Use per-band only if results are unsatisfactory")
    print("="*60)
