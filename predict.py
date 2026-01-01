#!/usr/bin/env python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")

class Predictor:
    def setup(self):
        # ✅ SINGLE MODEL REFERENCE - Your new standalone repo
        model_id = "Budhi1997/budhigpd-raw-3b-final"
        
        print(f"Loading tokenizer and model from: {model_id}")
        
        # Load everything from your single repo
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print("Model loaded successfully!")

    def predict(self, prompt, max_length=128, temperature=0.8, top_p=0.95, repetition_penalty=1.1):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            return response
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"Error: {str(e)[:100]}"
