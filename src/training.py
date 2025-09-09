"""
Training module for fine-tuning Llama models on coding datasets.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Optional
import json
import os
from .data_processing import CodeDataProcessor


class CodeLlamaTrainer:
    """Trainer for fine-tuning CodeLlama on bug detection tasks."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
        """Initialize the trainer with a base model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.data_processor = CodeDataProcessor(model_name)
        
    def setup_model_and_tokenizer(self):
        """Setup the model and tokenizer for training."""
        print(f"Loading model and tokenizer: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with LoRA for efficient fine-tuning
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("✅ Model and tokenizer setup complete!")
    
    def create_training_dataset(self) -> Dataset:
        """Create a training dataset for bug detection."""
        print("Creating training dataset...")
        
        # Load datasets
        datasets = self.data_processor.load_bug_detection_datasets()
        
        # Prepare training examples
        training_data = []
        
        # Create bug detection examples
        for dataset_name, dataset in datasets.items():
            print(f"Processing {dataset_name}...")
            
            count = 0
            for example in dataset:
                if count >= 1000:  # Limit for initial training
                    break
                
                # Extract code based on dataset format
                if dataset_name == 'codesearchnet':
                    code = example.get('func_code_string', '')
                    docstring = example.get('func_documentation_string', '')
                elif dataset_name == 'code_debug':
                    code = example.get('output', '')
                    instruction = example.get('instruction', '')
                else:
                    continue
                
                if code and len(code.strip()) > 20:
                    # Create training examples
                    training_data.extend(self._create_bug_detection_examples(code))
                    count += 1
        
        print(f"Created {len(training_data)} training examples")
        return Dataset.from_pandas(pd.DataFrame(training_data))
    
    def _create_bug_detection_examples(self, code: str) -> List[Dict]:
        """Create training examples for bug detection from code."""
        examples = []
        
        # Analyze the original code
        syntax_errors = self.data_processor.detect_syntax_errors(code)
        common_bugs = self.data_processor.detect_common_bugs(code)
        
        # Create example for clean code analysis
        if not syntax_errors and not common_bugs:
            prompt = f"Analyze this Python code for bugs:\n\n{code}\n\nAnalysis:"
            response = "This code appears to be free of obvious bugs and follows good practices."
            
            examples.append({
                'input': prompt,
                'output': response,
                'code': code,
                'has_bugs': False
            })
        
        # Create examples for buggy code
        else:
            issues_description = []
            for error in syntax_errors:
                issues_description.append(f"Syntax error on line {error.get('line', '?')}: {error['message']}")
            
            for bug in common_bugs:
                issues_description.append(f"{bug['type']} on line {bug.get('line', '?')}: {bug['message']}")
            
            prompt = f"Analyze this Python code for bugs:\n\n{code}\n\nAnalysis:"
            response = "Issues found:\n" + "\n".join(f"- {desc}" for desc in issues_description)
            
            examples.append({
                'input': prompt,
                'output': response,
                'code': code,
                'has_bugs': True
            })
        
        # Create debugging examples
        if syntax_errors or common_bugs:
            debug_prompt = f"Debug this Python code that has issues:\n\n{code}\n\nDebugging help:"
            debug_response = self._generate_debug_response(code, syntax_errors + common_bugs)
            
            examples.append({
                'input': debug_prompt,
                'output': debug_response,
                'code': code,
                'task': 'debugging'
            })
        
        return examples
    
    def _generate_debug_response(self, code: str, issues: List[Dict]) -> str:
        """Generate a debugging response for training."""
        response_parts = []
        response_parts.append("Let me help you debug this code:")
        response_parts.append("")
        
        for i, issue in enumerate(issues, 1):
            response_parts.append(f"{i}. **{issue['type']}**:")
            response_parts.append(f"   Problem: {issue['message']}")
            
            if issue['type'] == 'SyntaxError':
                response_parts.append(f"   Solution: Fix the syntax error on line {issue.get('line', '?')}")
            elif issue['type'] == 'AssignmentInCondition':
                response_parts.append(f"   Solution: Change '=' to '==' for comparison on line {issue.get('line', '?')}")
            elif issue['type'] == 'BroadException':
                response_parts.append(f"   Solution: Use specific exception types instead of bare 'except:' on line {issue.get('line', '?')}")
            elif issue['type'] == 'StringConcatenation':
                response_parts.append(f"   Solution: Use f-strings instead of string concatenation on line {issue.get('line', '?')}")
            
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        def tokenize_function(examples):
            # Combine input and output for causal language modeling
            texts = []
            for inp, out in zip(examples['input'], examples['output']):
                # Format as instruction-following
                text = f"<s>[INST] {inp} [/INST] {out} </s>"
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=2048,
                return_tensors="pt"
            )
            
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, 
              output_dir: str = "./models/coding-assistant",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4):
        """Train the model on the bug detection dataset."""
        
        if self.model is None or self.tokenizer is None:
            self.setup_model_and_tokenizer()
        
        # Create training dataset
        train_dataset = self.create_training_dataset()
        tokenized_dataset = self.tokenize_dataset(train_dataset)
        
        # Split dataset
        train_size = int(0.9 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset, eval_dataset = torch.utils.data.random_split(
            tokenized_dataset, [train_size, eval_size]
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            fp16=True,
            dataloader_pin_memory=False,
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        print(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("✅ Training complete!")
        
        return trainer
    
    def evaluate_model(self, test_codes: List[str], output_file: str = "evaluation_results.json"):
        """Evaluate the trained model on test code examples."""
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return
        
        results = []
        
        for i, code in enumerate(test_codes):
            print(f"Evaluating example {i+1}/{len(test_codes)}")
            
            # Ground truth analysis
            syntax_errors = self.data_processor.detect_syntax_errors(code)
            common_bugs = self.data_processor.detect_common_bugs(code)
            
            # Model prediction
            prompt = f"Analyze this Python code for bugs:\n\n{code}\n\nAnalysis:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = generated[len(prompt):].strip()
            
            results.append({
                'code': code,
                'ground_truth_syntax_errors': len(syntax_errors),
                'ground_truth_common_bugs': len(common_bugs),
                'model_prediction': prediction,
                'has_issues': len(syntax_errors) + len(common_bugs) > 0
            })
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {output_file}")
        return results


def main():
    """Main function for training the coding assistant."""
    trainer = CodeLlamaTrainer()
    
    # Test codes for evaluation
    test_codes = [
        """
def divide_numbers(a, b):
    if b = 0:  # Bug: assignment instead of comparison
        return "Cannot divide by zero"
    return a / b
""",
        """
def process_list(items):
    result = []
    for item in items:
        try:
            processed = item.upper()
            result.append(processed)
        except:  # Bug: too broad exception
            print("Error processing item")
    return result
""",
        """
def calculate_total(prices):
    total = 0
    for price in prices:
        total += price
    return total
"""  # Clean code
    ]
    
    print("Starting training process...")
    
    # Train the model
    trainer.train(
        output_dir="./models/coding-assistant-v1",
        num_epochs=2,
        batch_size=2,
        learning_rate=2e-4
    )
    
    # Evaluate on test cases
    trainer.evaluate_model(test_codes, "evaluation_results.json")


if __name__ == "__main__":
    main()