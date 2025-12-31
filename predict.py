#!/usr/bin/env python
import os
import torch
from typing import Iterator
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ⚠️ MUST HAVE THESE ENVIRONMENT VARIABLES
os.environ["TRANSFORMERS_CACHE"] = "/src/.cache/huggingface"
os.environ["HF_HOME"] = "/src/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Predictor(BasePredictor):  # ⚠️ MUST HAVE THIS CLASS NAME
    def setup(self):
        """Load model once at startup"""
        print("🚀 Loading BudhiGPD 3B model...")
        
        # Your model ID
        model_id = "Budhi1997/budhi-3b/budhigpd-raw-3b-final_merged"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print("✅ Model loaded successfully!")
    
    def predict(
        self,
        prompt: str = Input(description="Your message to BudhiGPD", default="Hello, who are you?"),
        max_length: int = Input(description="Maximum tokens to generate", default=512, ge=50, le=2048),
        temperature: float = Input(description="Creativity (0.1-1.0)", default=0.7, ge=0.1, le=1.0),
        top_p: float = Input(description="Focus (0.0-1.0)", default=0.9, ge=0.0, le=1.0),
        repetition_penalty: float = Input(description="Avoid repetition (1.0-2.0)", default=1.05, ge=1.0, le=2.0),
        do_sample: bool = Input(description="Use sampling", default=True),
        stream: bool = Input(description="Stream response", default=False),
    ) -> Iterator[str]:
        """Run a single prediction"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        if stream:
            # Streaming mode
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=60.0
            )
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            for token in streamer:
                yield token
            
            thread.join()
            
        else:
            # Non-streaming mode
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            yield response

# Optional: Keep your test code but make sure it doesn't run when imported by Cog
if __name__ == "__main__":
    # This only runs when you execute the file directly, not when imported by Cog
    predictor = Predictor()
    predictor.setup()
    
    print("\n" + "="*60)
    print("TEST: Non-streaming mode")
    print("="*60)
    for response in predictor.predict("Hello, who are you?", stream=False):
        print(f"Response: {response}")
    
    print("\n" + "="*60)
    print("TEST: Streaming mode")
    print("="*60)
    for token in predictor.predict("Hello, who are you?", stream=True):
        print(token, end="", flush=True)
    print()
