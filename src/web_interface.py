"""
Web interface for the coding assistant using Gradio.
"""

import gradio as gr
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import CodingAssistant
from src.data_processing import CodeDataProcessor
import json


class CodingAssistantInterface:
    """Web interface for the coding assistant."""
    
    def __init__(self):
        self.assistant = None
        self.processor = CodeDataProcessor()
        self.model_loaded = False
    
    def load_model(self, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
        """Load the model for the assistant."""
        try:
            self.assistant = CodingAssistant(model_name)
            self.assistant.load_model(use_quantization=True)
            self.model_loaded = True
            return "✅ Model loaded successfully!"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def analyze_code_interface(self, code: str, use_ai: bool = False):
        """Interface function for code analysis."""
        if not code.strip():
            return "Please enter some code to analyze.", "", ""
        
        try:
            # Static analysis (always available)
            syntax_errors = self.processor.detect_syntax_errors(code)
            common_bugs = self.processor.detect_common_bugs(code)
            
            # Format results
            results = {
                'syntax_errors': syntax_errors,
                'common_bugs': common_bugs,
                'total_issues': len(syntax_errors) + len(common_bugs)
            }
            
            # Generate report
            report = self._format_analysis_report(results)
            
            # Generate suggestions
            suggestions = self._generate_static_suggestions(syntax_errors + common_bugs)
            
            # AI analysis (if model is loaded and requested)
            ai_analysis = ""
            if use_ai and self.model_loaded and self.assistant:
                try:
                    full_analysis = self.assistant.analyze_code(code)
                    ai_analysis = full_analysis.get('ai_analysis', 'No AI analysis available')
                except Exception as e:
                    ai_analysis = f"Error in AI analysis: {str(e)}"
            elif use_ai and not self.model_loaded:
                ai_analysis = "Model not loaded. Use the 'Load Model' button first."
            
            return report, suggestions, ai_analysis
            
        except Exception as e:
            return f"Error analyzing code: {str(e)}", "", ""
    
    def debug_code_interface(self, code: str, error_message: str = ""):
        """Interface function for debugging help."""
        if not self.model_loaded or not self.assistant:
            return "Model not loaded. Please load the model first."
        
        if not code.strip():
            return "Please enter some code to debug."
        
        try:
            debug_result = self.assistant.debug_code(code, error_message)
            return debug_result.get('debug_help', 'No debugging help available')
        except Exception as e:
            return f"Error generating debug help: {str(e)}"
    
    def _format_analysis_report(self, results: dict) -> str:
        """Format the analysis results into a readable report."""
        report = []
        report.append("# Code Analysis Report")
        report.append(f"**Total Issues Found:** {results['total_issues']}")
        report.append("")
        
        if results['syntax_errors']:
            report.append("## Syntax Errors (High Priority)")
            for error in results['syntax_errors']:
                line_info = f" (Line {error['line']})" if 'line' in error else ""
                report.append(f"- **{error['type']}**{line_info}: {error['message']}")
            report.append("")
        
        if results['common_bugs']:
            report.append("## Code Quality Issues")
            for bug in results['common_bugs']:
                line_info = f" (Line {bug['line']})" if 'line' in bug else ""
                severity = bug.get('severity', 'unknown').upper()
                report.append(f"- **{bug['type']}** [{severity}]{line_info}: {bug['message']}")
            report.append("")
        
        if not results['syntax_errors'] and not results['common_bugs']:
            report.append("✅ No obvious issues detected! Your code looks good.")
        
        return "\n".join(report)
    
    def _generate_static_suggestions(self, issues: list) -> str:
        """Generate suggestions based on static analysis."""
        if not issues:
            return "No specific suggestions needed. Your code looks good!"
        
        suggestions = []
        suggestions.append("# Suggested Fixes")
        suggestions.append("")
        
        for i, issue in enumerate(issues, 1):
            suggestions.append(f"## Fix {i}: {issue['type']}")
            
            if issue['type'] == 'SyntaxError':
                suggestions.append(f"**Problem:** {issue['message']}")
                if 'line' in issue:
                    suggestions.append(f"**Location:** Line {issue['line']}")
                suggestions.append("**Solution:** Fix the syntax error as indicated.")
            
            elif issue['type'] == 'AssignmentInCondition':
                suggestions.append("**Problem:** Using assignment (=) instead of comparison (==) in condition")
                suggestions.append(f"**Location:** Line {issue['line']}")
                suggestions.append("**Solution:** Change `=` to `==` for comparison")
            
            elif issue['type'] == 'BroadException':
                suggestions.append("**Problem:** Too broad exception handling")
                suggestions.append(f"**Location:** Line {issue['line']}")
                suggestions.append("**Solution:** Use specific exception types like `ValueError`, `TypeError`, etc.")
            
            elif issue['type'] == 'StringConcatenation':
                suggestions.append("**Problem:** Using string concatenation in print statement")
                suggestions.append(f"**Location:** Line {issue['line']}")
                suggestions.append("**Solution:** Use f-strings for better performance: `f'text {variable}'`")
            
            suggestions.append("")
        
        return "\n".join(suggestions)


def create_interface():
    """Create and return the Gradio interface."""
    
    interface = CodingAssistantInterface()
    
    with gr.Blocks(title="AI Coding Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🤖 AI Coding Assistant")
        gr.Markdown("Detect bugs and get debugging help for your Python code using Llama models.")
        
        with gr.Tab("Code Analysis"):
            with gr.Row():
                with gr.Column():
                    code_input = gr.Textbox(
                        label="Python Code",
                        placeholder="Paste your Python code here...",
                        lines=15,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button("Analyze Code", variant="primary")
                        use_ai_checkbox = gr.Checkbox(
                            label="Use AI Analysis (requires model loading)", 
                            value=False
                        )
                
                with gr.Column():
                    analysis_output = gr.Markdown(label="Analysis Report")
                    suggestions_output = gr.Markdown(label="Suggestions")
            
            ai_analysis_output = gr.Textbox(
                label="AI Analysis (Llama Model)",
                lines=8,
                max_lines=15,
                interactive=False
            )
        
        with gr.Tab("Debug Help"):
            with gr.Row():
                with gr.Column():
                    debug_code_input = gr.Textbox(
                        label="Code to Debug",
                        placeholder="Paste the problematic code here...",
                        lines=10
                    )
                    error_input = gr.Textbox(
                        label="Error Message (optional)",
                        placeholder="Paste any error messages here...",
                        lines=3
                    )
                    debug_btn = gr.Button("Get Debug Help", variant="primary")
                
                with gr.Column():
                    debug_output = gr.Textbox(
                        label="Debug Help",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
        
        with gr.Tab("Model Management"):
            gr.Markdown("## Model Loading")
            gr.Markdown("Load the Llama model for AI-powered analysis. This may take a few minutes and requires significant GPU memory.")
            
            model_name_input = gr.Textbox(
                label="Model Name",
                value="codellama/CodeLlama-7b-Python-hf",
                placeholder="Enter Hugging Face model name..."
            )
            
            load_model_btn = gr.Button("Load Model", variant="secondary")
            model_status = gr.Textbox(
                label="Model Status",
                value="Model not loaded",
                interactive=False
            )
        
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Tool
            
            This AI Coding Assistant helps you:
            - **Detect bugs** in your Python code
            - **Get debugging suggestions** using AI
            - **Improve code quality** with best practice recommendations
            - **Fix common issues** automatically
            
            ### Features:
            - 🔍 **Static Analysis**: Detects syntax errors and common bugs without AI
            - 🤖 **AI Analysis**: Uses Llama models for advanced code review
            - 🛠️ **Debug Help**: Get step-by-step debugging assistance
            - 📝 **Suggestions**: Actionable recommendations for fixes
            
            ### Models Used:
            - **CodeLlama**: Specialized for code understanding and generation
            - **Datasets**: Trained on coding datasets from Hugging Face
            
            ### Requirements:
            - GPU recommended for AI analysis
            - Model loading requires ~8GB+ GPU memory for 7B models
            """)
        
        # Event handlers
        analyze_btn.click(
            fn=interface.analyze_code_interface,
            inputs=[code_input, use_ai_checkbox],
            outputs=[analysis_output, suggestions_output, ai_analysis_output]
        )
        
        debug_btn.click(
            fn=interface.debug_code_interface,
            inputs=[debug_code_input, error_input],
            outputs=[debug_output]
        )
        
        load_model_btn.click(
            fn=interface.load_model,
            inputs=[model_name_input],
            outputs=[model_status]
        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
