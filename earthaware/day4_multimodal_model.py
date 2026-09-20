

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from day3_spectral_vit import SpectralViT

class MultispectralVLM(nn.Module):

    def __init__(self,
                 vision_config=None,
                 language_model_name="meta-llama/Llama-2-7b-hf",
                 use_lora=True,
                 lora_rank=8):
        
        super().__init__()

        if vision_config is None:
            vision_config = {
                'in_channels': 13,
                'image_size': 64,
                'patch_size': 8,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'use_spectral_attention': True
            }

        print("Initializing SpectralViT...")
        self.vision_encoder = SpectralViT(**vision_config)

        self.vision_embed_dim = vision_config['embed_dim']

        print(f"Loading language model: {language_model_name}")
        print("Note: Using smaller model for 4GB GPU constraint")

        try:
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                language_model_name,
                trust_remote_code=True
            )
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            language_hidden_size = self.language_model.config.hidden_size
            
        except Exception as e:
            print(f"Llama-2 not available: {e}")
            print("Falling back to GPT-2 for demonstration...")

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            self.language_model = AutoModelForCausalLM.from_pretrained("gpt2")
            language_hidden_size = self.language_model.config.n_embd

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Creating vision-language projection...")
        self.vision_projection = nn.Linear(
            self.vision_embed_dim,
            language_hidden_size
        )

        if use_lora:
            print(f"Applying LoRA (rank={lora_rank})...")

            model_type = self.language_model.config.model_type if hasattr(self.language_model.config, 'model_type') else ""
            
            print(f"Detected model type: {model_type}")
            
            if model_type == "gpt2":
                target_modules = ["c_attn"]  
                print("Using GPT-2 target modules: c_attn")
            elif "llama" in model_type or "llama" in language_model_name.lower():
                target_modules = ["q_proj", "v_proj"]
                print("Using Llama target modules: q_proj, v_proj")
            else:
                
                target_modules = ["c_attn"]  
                print(f"Using default target modules: {target_modules}")
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.language_model = get_peft_model(self.language_model, lora_config)
            self.language_model.print_trainable_parameters()

        nn.init.xavier_uniform_(self.vision_projection.weight)
        nn.init.zeros_(self.vision_projection.bias)
        
        print("[OK] Model initialized successfully")
    
    def encode_image(self, images, return_attention=False):

        if return_attention:
            visual_features, attn_weights = self.vision_encoder(
                images, 
                return_attention=True
            )
        else:
            visual_features = self.vision_encoder(images)
            attn_weights = None

        visual_embeddings = self.vision_projection(visual_features)

        if return_attention:
            return visual_embeddings, attn_weights
        else:
            return visual_embeddings
    
    def forward(self, images, input_ids, attention_mask=None, labels=None):
        
        batch_size = images.shape[0]

        visual_embeddings = self.encode_image(images)

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        combined_embeddings = torch.cat([visual_embeddings, text_embeddings], dim=1)

        num_visual_tokens = visual_embeddings.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        visual_attention_mask = torch.ones(
            batch_size, num_visual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        
        combined_attention_mask = torch.cat(
            [visual_attention_mask, attention_mask],
            dim=1
        )

        if labels is not None:
            
            visual_labels = torch.full(
                (batch_size, num_visual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, images, prompt_text, max_new_tokens=50, temperature=0.7):
        
        self.eval()
        
        with torch.no_grad():
            
            visual_embeddings = self.encode_image(images)

            if isinstance(prompt_text, str):
                prompt_text = [prompt_text]
            
            prompt_tokens = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True
            ).to(visual_embeddings.device)

            text_embeddings = self.language_model.get_input_embeddings()(
                prompt_tokens.input_ids
            )

            combined_embeddings = torch.cat(
                [visual_embeddings, text_embeddings],
                dim=1
            )

            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=torch.ones(
                    combined_embeddings.shape[:2],
                    device=combined_embeddings.device
                )
            )
            
            logits = outputs.logits

            next_token_logits = logits[:, -1, :] / temperature
            next_tokens = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1),
                num_samples=1
            )

            generated_text = self.tokenizer.decode(
                next_tokens[0],
                skip_special_tokens=True
            )
            
            return generated_text

if __name__ == "__main__":

    print("="*60)
    print("TESTING MULTISPECTRAL VISION-LANGUAGE MODEL")
    print("="*60)

    print("\nInitializing model...")
    model = MultispectralVLM(
        use_lora=True,
        lora_rank=8
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100*trainable_params/total_params:.2f}%")

    print("\nTesting forward pass...")
    
    batch_size = 1
    dummy_image = torch.randn(batch_size, 13, 64, 64)

    text = "This is a test caption."
    tokens = model.tokenizer(
        text,
        return_tensors="pt",
        padding=True
    )
    
    print(f"  Image shape: {dummy_image.shape}")
    print(f"  Text tokens: {tokens.input_ids.shape}")

    with torch.no_grad():
        outputs = model(
            dummy_image,
            tokens.input_ids,
            tokens.attention_mask
        )
    
    print(f"  Output logits shape: {outputs.logits.shape}")
    print(f"  [OK] Forward pass successful")

    print("\nTesting image encoding...")
    with torch.no_grad():
        visual_emb = model.encode_image(dummy_image)
    
    print(f"  Visual embeddings: {visual_emb.shape}")
    print(f"  [OK] Image encoding successful")

    if torch.cuda.is_available():
        print("\nTesting GPU...")
        try:
            model_gpu = model.cuda()
            image_gpu = dummy_image.cuda()
            tokens_gpu = {k: v.cuda() for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs_gpu = model_gpu(
                    image_gpu,
                    tokens_gpu['input_ids'],
                    tokens_gpu['attention_mask']
                )
            
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"  [OK] GPU forward pass successful")
            print(f"  GPU memory: {memory_allocated:.2f} GB")
            
            if memory_allocated > 3.5:
                print("  [WARN] Warning: High memory usage!")
                print("     Consider using gradient checkpointing")
            
        except RuntimeError as e:
            print(f"  [FAIL] GPU test failed: {e}")
            print("     Model may be too large for 4GB GPU")
            print("     Try: Smaller language model or more aggressive quantization")
    
    print("\n" + "="*60)
    print("[OK] MULTIMODAL MODEL TEST COMPLETE")
    print("="*60)
    
    print("\nNotes:")
    print("- If GPU memory is tight, use gradient checkpointing")
    print("- For training, use batch_size=1 with gradient accumulation")
    print("- Consider using GPT-2 instead of Llama-2 for 4GB constraint")
