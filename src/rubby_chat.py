"""
Rubby Ducky Chat Assistant - For rubber duck debugging conversations.
"""

import re
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


class RubbyChatBot:
    """A friendly chatbot for rubber duck debugging conversations."""
    
    def __init__(self):
        self.conversation_history = []
        self.code_context = ""
        self.analysis_results = {}
        
        # Use rule-based responses by default for better debugging assistance
        # The AI model can be enabled later for more advanced conversations
        self.chat_pipeline = False
        print("� Rubby Chat initialized with rule-based responses (optimized for debugging)")
    
    def set_code_context(self, code: str, language: str, analysis_results: Dict[str, Any] = None):
        """Set the current code being discussed."""
        self.code_context = code
        self.language = language
        self.analysis_results = analysis_results or {}
        
        # Add context to conversation
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
        # Add user message to history
        self.conversation_history.append({
            'role': 'user', 
            'content': user_message
        })
        
        # Generate response
        if self.chat_pipeline:
            response = self._generate_ai_response(user_message)
        else:
            response = self._generate_rule_based_response(user_message)
        
        # Add response to history
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def _generate_ai_response(self, user_message: str) -> str:
        """Generate response using AI model."""
        try:
            # Prepare conversation context
            chat_history_ids = None
            
            # Encode the new user input and add special tokens
            new_user_input_ids = self.tokenizer.encode(
                user_message + self.tokenizer.eos_token, 
                return_tensors='pt'
            )
            
            # Build conversation context
            if len(self.conversation_history) > 2:
                # Get last few messages for context
                recent_history = self.conversation_history[-4:]
                context = ""
                for msg in recent_history[:-1]:  # Exclude the current user message we just added
                    if msg['role'] == 'user':
                        context += f"User: {msg['content']}\n"
                    else:
                        context += f"Assistant: {msg['content']}\n"
                
                chat_history_ids = self.tokenizer.encode(
                    context + self.tokenizer.eos_token,
                    return_tensors='pt'
                )
            
            # Generate response
            with torch.no_grad():
                if chat_history_ids is not None:
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                else:
                    bot_input_ids = new_user_input_ids
                
                chat_history_ids = self.model.generate(
                    bot_input_ids,
                    max_length=min(200, bot_input_ids.shape[-1] + 50),
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                skip_special_tokens=True
            )
            
            # Clean up and add rubber duck personality
            response = self._add_duck_personality(response.strip())
            
            return response or self._generate_rule_based_response(user_message)
            
        except Exception as e:
            print(f"AI response error: {e}")
            return self._generate_rule_based_response(user_message)
    
    def _generate_rule_based_response(self, user_message: str) -> str:
        """Generate response using rule-based patterns."""
        message_lower = user_message.lower()
        
        # Specific Python syntax issues
        if 'semicolon' in message_lower and 'python' in message_lower:
            return "🦆 Aha! I see the issue! In Python, you don't need semicolons (;) at the end of lines like in Java, C++, or JavaScript. Python uses newlines and indentation instead!\n\n" \
                   "❌ Wrong: `print('hello');`\n" \
                   "✅ Correct: `print('hello')`\n\n" \
                   "Just remove all the semicolons from the end of your lines and your Python code should work perfectly! Can you show me your code so we can fix it together?"
        
        elif ('semicolon' in message_lower or ';' in user_message) and ('error' in message_lower or 'issue' in message_lower):
            return "🦆 Semicolon troubles! What programming language are you using? Different languages have different rules:\n\n" \
                   "• **Python**: No semicolons needed\n" \
                   "• **JavaScript/Java/C++**: Semicolons required\n" \
                   "• **Go**: Semicolons optional (added automatically)\n\n" \
                   "Can you tell me which language and show me the error message?"
        
        # Indentation issues  
        elif 'indentation' in message_lower or 'indent' in message_lower:
            return "🦆 Indentation is crucial in Python! Unlike other languages that use braces {}, Python uses indentation to define code blocks.\n\n" \
                   "Make sure you're using consistent spaces (4 spaces is standard) or tabs, but not both! Can you show me the specific lines that are giving you trouble?"
        
        # Syntax error patterns
        elif 'syntax error' in message_lower:
            return "🦆 Syntax errors are the easiest to fix once you know what to look for! Can you share the exact error message and the line of code it's pointing to? " \
                   "Let's walk through it character by character - often it's just a missing quote, bracket, or parenthesis!"
        
        # Rubber duck debugging responses  
        elif any(word in message_lower for word in ['bug', 'error', 'wrong', 'broken', 'issue']):
            if self.analysis_results:
                issues = self.analysis_results.get('issues', [])
                if issues:
                    return f"🦆 I see you're having trouble! I found {len(issues)} potential issues in your code. Let's walk through them step by step. What specific behavior are you seeing that's unexpected?"
                else:
                    return "🦆 Hmm, I didn't find any obvious issues in the code analysis. Can you tell me more about what's happening? Sometimes the best debugging happens when you explain the problem out loud!"
            else:
                return "🦆 Tell me more about the bug you're experiencing. What did you expect to happen, and what actually happened? Walking through it step by step often helps!"
        
        elif any(word in message_lower for word in ['explain', 'understand', 'how', 'why', 'what']):
            if self.code_context:
                return "🦆 Great question! I'm looking at your code right now. Can you point me to the specific part you'd like me to explain? Sometimes explaining code line by line helps you spot issues too!"
            else:
                return "🦆 I'd love to help explain! Can you share the specific code you're curious about? The act of describing what each part does often leads to insights!"
        
        elif any(word in message_lower for word in ['fix', 'solve', 'help']):
            return "🦆 Let's tackle this together! The rubber duck method works best when YOU do the explaining. Can you walk me through what your code is supposed to do, step by step? I'll listen and ask questions along the way!"
        
        elif any(word in message_lower for word in ['thanks', 'thank', 'good', 'great', 'awesome']):
            return "🦆 Quack! I'm happy to help! Remember, the best debugging often happens when you explain your problem out loud. Keep talking through your code - you've got this!"
        
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'start']):
            return "🦆 Hello there! I'm Rubby, your rubber duck debugging assistant! Share your code with me and let's talk through any issues you're having. What are you working on today?"
        
        elif 'logic' in message_lower or 'algorithm' in message_lower:
            return "🦆 Logic issues can be tricky! Try explaining to me what your algorithm is supposed to do in plain English, then we can compare that to what the code actually does. Where do you think the logic might be going wrong?"
        
        elif any(word in message_lower for word in ['performance', 'slow', 'fast', 'optimize']):
            return "🦆 Performance questions! Tell me about where you think the bottleneck might be. Have you identified which part of your code is running slowly? Sometimes explaining the data flow helps spot inefficiencies!"
        
        elif any(word in message_lower for word in ['test', 'testing']):
            return "🦆 Testing is so important! What kind of test cases have you tried? Walk me through what inputs you're testing and what outputs you expect. Edge cases often reveal bugs!"
        
        else:
            # More helpful generic responses based on context
            if 'python' in message_lower:
                responses = [
                    "🦆 Python questions are my favorite! What specific part of your Python code are you curious about?",
                    "🦆 Let's talk Python! Can you share the code you're working with so I can help better?",
                    "🦆 Python can be tricky sometimes! What's the specific issue you're facing?",
                ]
            elif any(lang in message_lower for lang in ['javascript', 'java', 'c++', 'rust', 'go']):
                responses = [
                    "🦆 Great choice of language! What specific problem are you trying to solve?",
                    "🦆 I'd love to help with your code! Can you share what you're working on?",
                    "🦆 Tell me more about what you're trying to accomplish with your code!",
                ]
            else:
                responses = [
                    "🦆 I'm here to help debug! Can you share your code and describe what issue you're facing?",
                    "🦆 Let's work through this together! What specific problem are you trying to solve?",
                    "🦆 I'm listening! The more details you can share about your code issue, the better I can help.",
                    "🦆 Rubber duck debugging works best when you explain your problem step by step. What's going on?",
                    "🦆 I'm ready to help! Can you describe what your code is supposed to do vs what it's actually doing?"
                ]
            import random
            return random.choice(responses)
    
    def _add_duck_personality(self, response: str) -> str:
        """Add rubber duck personality to responses."""
        if not response:
            return "🦆 Quack! I'm here to help! What would you like to discuss about your code?"
        
        # Add duck emoji if not present
        if '🦆' not in response:
            response = "🦆 " + response
        
        # Ensure encouraging tone
        if not any(word in response.lower() for word in ['!', 'great', 'good', 'awesome', 'excellent']):
            if response.endswith('.'):
                response = response[:-1] + "!"
        
        return response
    
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
            "🧪 Create simple test cases to isolate the issue",
            "📊 Add print statements to see variable values",
            "🤔 Consider edge cases and boundary conditions",
            "⏰ Take a break and come back with fresh eyes"
        ]
