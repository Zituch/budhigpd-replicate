#!/usr/bin/env python
import os
import time
import torch
from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ⚠️ ADD THESE ENVIRONMENT VARIABLES - REQUIRED FOR Replicate
os.environ["TRANSFORMERS_CACHE"] = "/src/.cache/huggingface"
os.environ["HF_HOME"] = "/src/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache
_model = None
_tokenizer = None

def setup():
    """Load model once at startup"""
    global _model, _tokenizer
    
    if _model is not None:
        return
    
    print("🚀 Loading BudhiGPD 3B model...")
    start_time = time.time()
    
    # ✅ Direct path to your actual model folder
    model_id = "Budhi1997/budhi-3b/budhigpd-raw-3b-final_merged"
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add pad token if missing
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    
    # Load model with bfloat16 for universal GPU compatibility
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # ✅ Universal GPU compatibility
        device_map="auto",
        trust_remote_code=True,
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")

def predict(
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    do_sample: bool = True,
    stream: bool = False,
) -> Iterator[str]:
    """Main prediction function with streaming support"""
    
    # Ensure model is loaded
    if _model is None:
        setup()
    
    # Tokenize input
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(_model.device)
    
    if stream:
        # ✅ Streaming mode: tokens appear as they're generated
        streamer = TextIteratorStreamer(
            _tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0  # Timeout for streaming
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
            use_cache=True,  # ✅ Critical for GPU speed
        )
        
        # Start generation in separate thread
        thread = Thread(target=_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens as they become available
        for token in streamer:
            yield token
        
        # Wait for thread to complete
        thread.join()
        
    else:
        # ✅ Non-streaming mode: wait for complete response
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
                use_cache=True,  # ✅ Critical for GPU speed
            )
        
        # Decode complete response
        response = _tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        yield response

if __name__ == "__main__":
    # Test both modes
    setup()
    
    print("\n" + "="*60)
    print("TEST 1: Non-streaming mode (complete response)")
    print("="*60)
    for response in predict("Hello, who are you?", stream=False):
        print(f"Response: {response}")
        print(f"Length: {len(response.split())} words")
    
    print("\n" + "="*60)
    print("TEST 2: Streaming mode (tokens appear gradually)")
    print("="*60)
    for token in predict("Hello, who are you?", stream=True):
        print(token, end="", flush=True)
    print()  # Newline at end
