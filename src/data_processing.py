"""
Data processing module for the coding assistant.
Handles loading and preprocessing of coding datasets from Hugging Face.
"""

import ast
import re
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset, Dataset
import pandas as pd
import torch
from transformers import AutoTokenizer


class CodeDataProcessor:
    """Processes coding datasets for bug detection and debugging."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
        """Initialize the data processor with a specific model tokenizer."""
        self.model_name = model_name
        self.tokenizer = None
        self.max_length = 2048
        
    def load_tokenizer(self):
        """Load the tokenizer for the specified model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Loaded tokenizer for {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            raise
    
    def load_bug_detection_datasets(self) -> Dict[str, Dataset]:
        """Load relevant datasets from Hugging Face for bug detection."""
        datasets = {}
        
        try:
            # CodeSearchNet for general code examples
            print("Loading CodeSearchNet dataset...")
            datasets['codesearchnet'] = load_dataset(
                "code_search_net", 
                "python", 
                split="train[:5000]"  # Sample for initial testing
            )
            
            # Code debugging dataset
            print("Loading code debugging dataset...")
            datasets['code_debug'] = load_dataset(
                "sahil2801/CodeAlpaca-20k",
                split="train[:2000]"
            )
            
            # Python code quality dataset
            print("Loading Python code examples...")
            datasets['python_code'] = load_dataset(
                "codeparrot/github-code",
                streaming=True,
                languages=["Python"],
                split="train"
            )
            
            print(f"✅ Loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return {}
    
    def detect_syntax_errors(self, code: str) -> List[Dict]:
        """Detect syntax errors in Python code using AST."""
        errors = []
        
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append({
                'type': 'SyntaxError',
                'line': e.lineno,
                'offset': e.offset,
                'message': e.msg,
                'severity': 'high'
            })
        except Exception as e:
            errors.append({
                'type': 'ParseError',
                'message': str(e),
                'severity': 'medium'
            })
        
        return errors
    
    def detect_common_bugs(self, code: str) -> List[Dict]:
        """Detect common Python bugs using pattern matching."""
        bugs = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for common issues
            if '==' in line and 'if' in line and '=' in line.replace('==', ''):
                bugs.append({
                    'type': 'AssignmentInCondition',
                    'line': i,
                    'message': 'Possible assignment (=) instead of comparison (==)',
                    'severity': 'medium'
                })
            
            if 'except:' in line or 'except Exception:' in line:
                bugs.append({
                    'type': 'BroadException',
                    'line': i,
                    'message': 'Too broad exception handling',
                    'severity': 'low'
                })
            
            if re.search(r'print\(.*\+.*\)', line):
                bugs.append({
                    'type': 'StringConcatenation',
                    'line': i,
                    'message': 'Consider using f-strings instead of string concatenation',
                    'severity': 'low'
                })
        
        return bugs
    
    def prepare_training_data(self, datasets: Dict[str, Dataset]) -> Dataset:
        """Prepare data for training the bug detection model."""
        training_examples = []
        
        for dataset_name, dataset in datasets.items():
            print(f"Processing {dataset_name}...")
            
            for example in dataset:
                if dataset_name == 'codesearchnet':
                    code = example.get('func_code_string', '')
                elif dataset_name == 'code_debug':
                    code = example.get('output', '')
                else:
                    continue
                
                if code and len(code.strip()) > 10:
                    # Detect issues in the code
                    syntax_errors = self.detect_syntax_errors(code)
                    common_bugs = self.detect_common_bugs(code)
                    
                    # Create training example
                    issues = syntax_errors + common_bugs
                    
                    training_examples.append({
                        'code': code,
                        'has_bugs': len(issues) > 0,
                        'issues': issues,
                        'source': dataset_name
                    })
        
        print(f"✅ Prepared {len(training_examples)} training examples")
        return Dataset.from_pandas(pd.DataFrame(training_examples))
    
    def tokenize_data(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for model training."""
        if self.tokenizer is None:
            self.load_tokenizer()
        
        def tokenize_function(examples):
            # Create input text for the model
            inputs = []
            for code in examples['code']:
                prompt = f"Analyze this Python code for bugs:\n\n{code}\n\nIssues found:"
                inputs.append(prompt)
            
            # Tokenize
            tokenized = self.tokenizer(
                inputs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


if __name__ == "__main__":
    # Test the data processor
    processor = CodeDataProcessor()
    
    # Test code analysis
    test_code = """
def example_function(x, y):
    if x = 5:  # Bug: assignment instead of comparison
        result = x + y
    return result

try:
    value = example_function(3, 4)
    print("Result: " + str(value))  # Improvement: use f-string
except:  # Bug: too broad exception
    print("Error occurred")
"""
    
    print("Testing syntax error detection:")
    syntax_errors = processor.detect_syntax_errors(test_code)
    for error in syntax_errors:
        print(f"  {error}")
    
    print("\nTesting common bug detection:")
    common_bugs = processor.detect_common_bugs(test_code)
    for bug in common_bugs:
        print(f"  {bug}")