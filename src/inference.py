"""
Inference module for the coding assistant.
Handles bug detection and debugging suggestions using Llama models.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from typing import List, Dict, Optional
import ast
import re
from .data_processing import CodeDataProcessor


class CodingAssistant:
    """Main coding assistant class for bug detection and debugging."""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
        """Initialize the coding assistant with a Llama model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.data_processor = CodeDataProcessor(model_name)
        
    def load_model(self, use_quantization: bool = True):
        """Load the Llama model for inference."""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Configure quantization for efficiency
            if use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def analyze_code(self, code: str) -> Dict:
        """Analyze code for bugs and provide debugging suggestions."""
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Basic syntax and pattern analysis
        syntax_errors = self.data_processor.detect_syntax_errors(code)
        common_bugs = self.data_processor.detect_common_bugs(code)
        
        # Generate AI-powered analysis
        ai_analysis = self._generate_ai_analysis(code)
        
        # Combine results
        analysis_result = {
            'code': code,
            'syntax_errors': syntax_errors,
            'common_bugs': common_bugs,
            'ai_analysis': ai_analysis,
            'severity_score': self._calculate_severity_score(syntax_errors + common_bugs),
            'suggestions': self._generate_suggestions(syntax_errors + common_bugs, ai_analysis)
        }
        
        return analysis_result
    
    def _generate_ai_analysis(self, code: str) -> str:
        """Generate AI-powered code analysis using the Llama model."""
        prompt = f"""<s>[INST] You are an expert Python developer and code reviewer. Analyze the following Python code for potential bugs, issues, and improvements. Provide specific suggestions for fixing any problems you find.

Code to analyze:
```python
{code}
```

Please provide:
1. Any bugs or errors you find
2. Code quality issues
3. Performance improvements
4. Best practice recommendations
5. Specific fixes for each issue

Analysis: [/INST]"""

        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the assistant's response
            if "[/INST]" in generated_text:
                analysis = generated_text.split("[/INST]")[-1].strip()
            else:
                analysis = generated_text
            
            return analysis
            
        except Exception as e:
            return f"Error generating AI analysis: {str(e)}"
    
    def _calculate_severity_score(self, issues: List[Dict]) -> float:
        """Calculate a severity score based on detected issues."""
        if not issues:
            return 0.0
        
        severity_weights = {'high': 3, 'medium': 2, 'low': 1}
        total_score = sum(severity_weights.get(issue.get('severity', 'low'), 1) for issue in issues)
        max_possible = len(issues) * 3
        
        return total_score / max_possible if max_possible > 0 else 0.0
    
    def _generate_suggestions(self, issues: List[Dict], ai_analysis: str) -> List[str]:
        """Generate actionable suggestions for fixing issues."""
        suggestions = []
        
        # Add suggestions based on detected issues
        for issue in issues:
            if issue['type'] == 'SyntaxError':
                suggestions.append(f"Fix syntax error on line {issue['line']}: {issue['message']}")
            elif issue['type'] == 'AssignmentInCondition':
                suggestions.append(f"Line {issue['line']}: Change assignment (=) to comparison (==) in condition")
            elif issue['type'] == 'BroadException':
                suggestions.append(f"Line {issue['line']}: Use specific exception types instead of broad exception handling")
            elif issue['type'] == 'StringConcatenation':
                suggestions.append(f"Line {issue['line']}: Use f-strings for better performance and readability")
            elif issue['type'] == 'TypeMismatchConcatenation':
                suggestions.append(f"Line {issue['line']}: Convert number to string or use f-string: str(number) or f'text {{number}}'")
            elif issue['type'] == 'DivisionByZero':
                suggestions.append(f"Line {issue['line']}: Add check for zero before division to prevent ZeroDivisionError")
            elif issue['type'] == 'PossibleUndefinedVariable':
                suggestions.append(f"Line {issue['line']}: Ensure variable is defined before use")
        
        # Add AI-generated suggestions
        if ai_analysis and "Error generating" not in ai_analysis:
            suggestions.append("AI Analysis: " + ai_analysis[:200] + "..." if len(ai_analysis) > 200 else ai_analysis)
        
        return suggestions
    
    def fix_code(self, code: str) -> str:
        """Attempt to automatically fix common issues in code."""
        fixed_code = code
        
        # Fix common patterns
        lines = fixed_code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix assignment in conditions (simple cases)
            if 'if' in line and re.search(r'if\s+\w+\s*=\s*\w+', line):
                line = re.sub(r'(\w+)\s*=\s*(\w+)', r'\1 == \2', line)
            
            # Fix string concatenation to f-strings (simple cases)
            if 'print(' in line and '+' in line:
                # This is a simplified fix - in practice, you'd want more sophisticated parsing
                pass
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def debug_code(self, code: str, error_message: str = "") -> Dict:
        """Provide debugging help for code with errors."""
        prompt = f"""<s>[INST] You are a debugging expert. Help debug this Python code that has an error.

Code:
```python
{code}
```

Error message (if any): {error_message}

Please provide:
1. Explanation of what's wrong
2. Step-by-step debugging approach
3. Fixed version of the code
4. Prevention tips for similar issues

Debugging help: [/INST]"""

        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            if "[/INST]" in generated_text:
                debug_help = generated_text.split("[/INST]")[-1].strip()
            else:
                debug_help = generated_text
            
            return {
                'code': code,
                'error_message': error_message,
                'debug_help': debug_help,
                'analysis': self.analyze_code(code)
            }
            
        except Exception as e:
            return {
                'code': code,
                'error_message': error_message,
                'debug_help': f"Error generating debug help: {str(e)}",
                'analysis': self.analyze_code(code)
            }


def main():
    """Test the coding assistant."""
    assistant = CodingAssistant()
    
    # Test code with bugs
    test_code = """
def calculate_average(numbers):
    if len(numbers) = 0:  # Bug: assignment instead of comparison
        return 0
    
    total = 0
    for num in numbers:
        total += num
    
    try:
        average = total / len(numbers)
        print("Average is: " + str(average))  # Improvement: use f-string
        return average
    except:  # Bug: too broad exception
        print("Error calculating average")
        return None
"""
    
    print("Testing coding assistant...")
    print("=" * 50)
    
    # Note: In a real scenario, you would call assistant.load_model() first
    # For testing without GPU, we'll just test the static analysis
    
    # Test static analysis
    processor = CodeDataProcessor()
    syntax_errors = processor.detect_syntax_errors(test_code)
    common_bugs = processor.detect_common_bugs(test_code)
    
    print("Syntax Errors:")
    for error in syntax_errors:
        print(f"  - {error}")
    
    print("\nCommon Bugs:")
    for bug in common_bugs:
        print(f"  - {bug}")


if __name__ == "__main__":
    main()