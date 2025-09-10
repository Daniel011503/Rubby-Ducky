"""
Rubby Duck Chatbot - Interactive rubber duck debugging assistant
"""

import re
import random
from typing import Dict, List, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None


class RubbyChatBot:
    """A friendly chatbot for rubber duck debugging conversations."""
    
    def __init__(self, use_ai_model=False):
        self.conversation_history = []
        self.code_context = ""
        self.analysis_results = {}
        
        if use_ai_model and TRANSFORMERS_AVAILABLE:
            try:
                # Try Phi-3-mini first - lightweight, fast, coding-optimized
                print("🦆 Loading Phi-3-mini model for fast AI debugging conversations...")
                model_name = "microsoft/Phi-3-mini-4k-instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.chat_pipeline = True
                print("🤖 Rubby Chat initialized with Phi-3-mini (fast & coding-optimized)")
                
            except Exception as e:
                print(f"⚠️ Could not load Phi-3-mini: {e}")
                try:
                    # Fallback to CodeGemma-2B - even lighter
                    print("🦆 Trying CodeGemma-2B as fallback...")
                    model_name = "google/codegemma-2b-it"
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.chat_pipeline = True
                    print("🤖 Rubby Chat initialized with CodeGemma-2B (lightweight coding model)")
                    
                except Exception as e2:
                    print(f"⚠️ Could not load CodeGemma: {e2}")
                    print("💬 Using rule-based responses (still excellent for debugging!)")
                    self.chat_pipeline = False
        else:
            self.chat_pipeline = False
            print("🦆 Rubby Chat initialized with rule-based responses")
    
    def set_code_context(self, code: str, language: str, analysis_results: Dict[str, Any] = None):
        """Set the current code being discussed."""
        self.code_context = code
        self.language = language
        self.analysis_results = analysis_results or {}
        
        context_msg = f"I'm looking at your {language} code. "
        if analysis_results:
            if analysis_results.get('prediction') == 'clean':
                context_msg += f"It looks pretty good with {analysis_results.get('confidence', 0):.1%} confidence! "
            else:
                context_msg += f"I found some potential issues. "
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': context_msg + "What would you like to discuss about it? 🦆"
        })
    
    def chat(self, user_message: str) -> str:
        """Have a conversation with the user about their code."""
        self.conversation_history.append({
            'role': 'user', 
            'content': user_message
        })
        
        if self.chat_pipeline:
            response = self._generate_ai_response(user_message)
        else:
            response = self._generate_rule_based_response(user_message)
        
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response

    def _generate_ai_response(self, user_message: str) -> str:
        """Generate response using AI model."""
        try:
            conversation_context = ""
            if self.code_context:
                conversation_context += f"Code context ({self.language}):\n```\n{self.code_context[:500]}...\n```\n\n"
            
            recent_messages = self.conversation_history[-4:] if len(self.conversation_history) > 4 else self.conversation_history[:-1]
            for msg in recent_messages:
                if msg['role'] == 'user':
                    conversation_context += f"User: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    conversation_context += f"Assistant: {msg['content']}\n"
            
            prompt = f"""You are Rubby, a friendly rubber duck debugging assistant.

{conversation_context}

User: {user_message}

Respond as Rubby. Be helpful and encouraging. Start with 🦆."""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            if not response.startswith('🦆'):
                response = '🦆 ' + response
            
            return response
            
        except Exception as e:
            print(f"AI response error: {e}")
            return self._generate_rule_based_response(user_message)

    def _generate_rule_based_response(self, user_message: str) -> str:
        """Generate response using rule-based patterns."""
        user_lower = user_message.lower()
        
        # Python semicolon issue
        if any(keyword in user_lower for keyword in ['semicolon', ';', 'python']) and \
           any(keyword in user_lower for keyword in ['error', 'problem', 'wrong', 'issue']):
            return "🦆 Aha! I see the issue! In Python, you don't need semicolons at the end of lines. " \
                   "Unlike languages like Java or C++, Python uses indentation and line breaks to define code structure. " \
                   "Try removing those semicolons and your code should work perfectly! " \
                   "Is there anything specific about the error message you'd like me to explain?"
        
        # Syntax error patterns
        if any(keyword in user_lower for keyword in ['syntax error', 'syntaxerror', 'invalid syntax']):
            return "🦆 Syntax errors are like typos in code! Let's debug this step by step:\n" \
                   "1. Check for missing parentheses, brackets, or quotes\n" \
                   "2. Look at the line number in the error message\n" \
                   "3. Sometimes the real error is actually on the line BEFORE the reported line\n" \
                   "4. Check your indentation - Python is picky about that!\n" \
                   "Can you share the exact error message and the line it's pointing to?"
        
        # Greeting responses
        if any(greeting in user_lower for greeting in ['hello', 'hi', 'hey', 'help']):
            return "🦆 Hello there! I'm Rubby, your friendly rubber duck debugging companion! " \
                   "I'm here to help you work through any coding problems. Just explain what you're " \
                   "working on and what issues you're facing!"
        
        # Generic fallback
        if self.code_context:
            responses = [
                "🦆 I'm looking at your code! Can you tell me more about what specific issue you're seeing?",
                "🦆 Let's debug this together! What behavior are you expecting vs. what's actually happening?",
                "🦆 Perfect! I have your code context. Walk me through what you think should happen step by step.",
            ]
        else:
            responses = [
                "🦆 I'm here to help debug! Can you share your code and describe what issue you're facing?",
                "🦆 Let's work through this together! What specific problem are you trying to solve?",
                "🦆 I'm listening! The more details you can share about your code issue, the better I can help.",
            ]
        
        return random.choice(responses)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the full conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.code_context = ""
        self.analysis_results = {}
    
    def get_debugging_suggestions(self) -> List[str]:
        """Get general rubber duck debugging suggestions."""
        return [
            "🦆 Explain what your code is supposed to do, line by line",
            "🔍 Walk through the execution with sample data",
            "❓ Ask yourself: What did I expect vs. what actually happened?",
            "🎯 Identify the smallest piece that doesn't work as expected",
            "📝 Write down your assumptions and test them",
            "🔄 Try explaining the problem to someone else (or me!)",
            "🧪 Test with the simplest possible input first",
            "🔬 Use print statements to see what's happening",
            "📚 Check the documentation for functions you're using",
            "🎭 Step through your code line by line in a debugger"
        ]
