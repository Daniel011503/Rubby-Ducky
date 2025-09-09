"""
Main runner for Rubby Ducky - The Rubber Duck Debugging Assistant.
"""

import argparse
import sys
import os
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.multi_language_processor import MultiLanguageCodeProcessor


def run_analysis(code_file: str, model_name: str = "microsoft/codebert-base", language: str = "python"):
    """Run code analysis on a file."""
    print(f"Analyzing {language} code in: {code_file}")
    
    # Read code file
    with open(code_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Use multi-language processor with rule engine and ML capabilities for all languages
    processor = MultiLanguageCodeProcessor(language, model_name, use_ml_model=True)
    
    try:
        # Use ML-enhanced analysis for all languages with rule engine
        print(f"🤖 Using AI-enhanced analysis for multi-language detection...")
        
        # Get AI analysis with ML models and rule-based detection
        if processor.use_ml_model:
            ml_result = processor.predict_bugs_ml(code)
            
            # Get rule-based analysis first to inform AI classification
            syntax_errors = processor.detect_syntax_errors(code, language)
            common_bugs = processor.detect_common_bugs(code, language)
            
            # Calculate issue severity
            high_severity_count = len(syntax_errors)
            medium_severity_count = sum(1 for bug in common_bugs if bug.get('severity') == 'medium')
            low_severity_count = sum(1 for bug in common_bugs if bug.get('severity') == 'low')
            
            # Intelligent classification combining ML and rule-based findings
            if high_severity_count > 0:
                # Syntax errors = definitely buggy
                final_prediction = 'buggy'
                final_confidence = min(0.9, 0.7 + (high_severity_count * 0.1))
            elif medium_severity_count > 5:
                # Many medium issues = likely buggy
                final_prediction = 'buggy'
                final_confidence = min(0.8, 0.6 + (medium_severity_count * 0.03))
            elif low_severity_count > 0 and medium_severity_count <= 1:
                # Only minor issues or at most 1 medium issue = clean
                final_prediction = 'clean'
                final_confidence = max(0.85, 1.0 - (low_severity_count * 0.02) - (medium_severity_count * 0.05))
            elif high_severity_count == 0 and medium_severity_count == 0 and low_severity_count == 0:
                # No issues found = clean unless ML is very confident
                if ml_result['prediction'] == 'buggy' and ml_result['confidence'] > 0.7:
                    final_prediction = 'buggy'
                    final_confidence = ml_result['confidence']
                else:
                    final_prediction = 'clean'
                    final_confidence = max(0.90, 1.0 - ml_result['buggy_probability'])
            else:
                # Use ML prediction for borderline cases
                final_prediction = ml_result['prediction']
                final_confidence = ml_result['confidence']
            
            ai_insights = {
                'prediction': final_prediction,
                'confidence': final_confidence,
                'ml_available': True
            }
        else:
            ai_insights = {'ml_available': False}
            # Get rule-based analysis
            syntax_errors = processor.detect_syntax_errors(code, language)
            common_bugs = processor.detect_common_bugs(code, language)
        
        print("\n" + "="*50)
        print(f"{language.upper()} AI-ENHANCED CODE ANALYSIS REPORT")
        print("="*50)
        
        print(f"\nFile: {code_file}")
        print(f"Language: {language}")
        print(f"Lines of code: {len(code.splitlines())}")
        
        # Display AI insights
        if ai_insights.get('ml_available'):
            print(f"🤖 AI Analysis: {ai_insights['prediction'].title()} (confidence: {ai_insights['confidence']:.1%})")
        
        # Calculate severity score
        total_issues = len(syntax_errors) + len(common_bugs)
        severity_score = 0
        for error in syntax_errors:
            severity_score += 3
        for bug in common_bugs:
            severity_weights = {'high': 3, 'medium': 2, 'low': 1}
            severity_score += severity_weights.get(bug.get('severity', 'low'), 1)
        max_possible = total_issues * 3 if total_issues > 0 else 1
        severity_percentage = (severity_score / max_possible) * 100
        print(f"📊 Severity Score: {severity_percentage:.0f}%")
        
        if syntax_errors:
            print(f"\n🚨 SYNTAX ERRORS ({len(syntax_errors)}):")
            for error in syntax_errors:
                print(f"  - Line {error.get('line', '?')}: {error['message']}")
                if error.get('suggestion'):
                    print(f"    💡 Suggestion: {error['suggestion']}")
        
        if common_bugs:
            print(f"\n⚠️ CODE QUALITY ISSUES ({len(common_bugs)}):")
            for bug in common_bugs:
                severity = bug.get('severity', 'unknown').upper()
                print(f"  - [{severity}] Line {bug.get('line', '?')}: {bug['message']}")
                if bug.get('suggestion'):
                    print(f"    💡 Suggestion: {bug['suggestion']}")
        
        if not syntax_errors and not common_bugs:
            if ai_insights.get('prediction') == 'clean' or not ai_insights.get('ml_available'):
                print("\n✅ No obvious issues detected! Your code looks good.")
            else:
                print("\n⚠️ AI detected potential issues, but no specific problems found by static analysis.")
        
    except Exception as e:
        print(f"⚠️ Error during analysis: {e}")
        print("Falling back to basic pattern detection...")
        
        # Basic fallback without ML
        try:
            processor_fallback = MultiLanguageCodeProcessor(language, model_name, use_ml_model=False)
            syntax_errors = processor_fallback.detect_syntax_errors(code, language)
            common_bugs = processor_fallback.detect_common_bugs(code, language)
            
            print("\n" + "="*50)
            print(f"{language.upper()} BASIC ANALYSIS REPORT")
            print("="*50)
            
            print(f"\nFile: {code_file}")
            print(f"Language: {language}")
            print(f"Lines of code: {len(code.splitlines())}")
            
            if syntax_errors or common_bugs:
                if syntax_errors:
                    print(f"\n🚨 SYNTAX ERRORS ({len(syntax_errors)}):")
                    for error in syntax_errors:
                        print(f"  - Line {error.get('line', '?')}: {error['message']}")
                
                if common_bugs:
                    print(f"\n⚠️ CODE QUALITY ISSUES ({len(common_bugs)}):")
                    for bug in common_bugs:
                        severity = bug.get('severity', 'unknown').upper()
                        print(f"  - [{severity}] Line {bug.get('line', '?')}: {bug['message']}")
            else:
                print("\n✅ No obvious issues detected! Your code looks good.")
                
        except Exception as fallback_error:
            print(f"❌ Analysis failed: {fallback_error}")
            return None


def run_training(output_dir: str, epochs: int = 3, languages: List[str] = None, train_ml: bool = False, load_datasets: bool = False):
    """Run model training for multiple languages with ML capabilities."""
    if languages is None:
        languages = ["python"]
    
    print(f"🚀 Starting training process...")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Languages: {languages}")
    print(f"ML Training: {train_ml}")
    print(f"Load Datasets: {load_datasets}")
    
    results = []
    
    if train_ml or load_datasets:
        print("\n" + "="*60)
        print("🤖 ML-BASED BUG DETECTION TRAINING")
        print("="*60)
        
        for lang in languages:
            print(f"\n🔄 Processing {lang.upper()}...")
            
            try:
                # Initialize ML processor
                processor = MultiLanguageCodeProcessor(lang, use_ml_model=True)
                
                if load_datasets:
                    print(f"\n📊 Loading datasets for {lang}...")
                    # Note: Dataset loading functionality needs to be implemented
                    print(f"⚠️ Dataset loading for {lang} is not yet implemented.")
                    print("This would require integration with bug detection datasets like:")
                    print("  - Defects4J for Java")
                    print("  - CodeXGLUE for multi-language")
                    print("  - GitHub bug fix commits")
                
                if train_ml:
                    print(f"\n🎯 Training ML bug classifier for {lang}...")
                    print(f"⚠️ ML training for {lang} is not yet implemented.")
                    print("This would require:")
                    print("  - Curated training datasets")
                    print("  - Fine-tuning CodeBERT models")
                    print("  - Validation and evaluation")
                    
                results.append(f"ℹ️ {lang}: Training setup acknowledged (implementation pending)")
                
            except Exception as e:
                error_msg = f"❌ {lang}: Error - {e}"
                print(error_msg)
                results.append(error_msg)
    else:
        print("\n" + "="*60)
        print("ℹ️ TRAINING INFORMATION")
        print("="*60)
        print("To implement full training capabilities, you would need:")
        print("1. Bug detection datasets (Defects4J, CodeXGLUE)")
        print("2. Fine-tuning pipeline for CodeBERT")
        print("3. Training/validation/test splits")
        print("4. Model evaluation metrics")
        print("\nCurrently using pre-trained CodeBERT models.")
        
        results.append("ℹ️ Training information provided")
    
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    for result in results:
        print(result)
    
    return {"status": "completed", "results": results}


def run_web_interface(port: int = 8501):
    """Launch the Streamlit web interface."""
    print(f"🦆 Launching Rubby Ducky web interface on port {port}")
    print(f"🌐 Access at: http://localhost:{port}")
    
    # Import subprocess to run streamlit
    import subprocess
    import sys
    
    # Get the python executable path
    python_exe = sys.executable
    
    # Run streamlit with the correct app file
    try:
        subprocess.run([
            python_exe, "-m", "streamlit", "run", 
            "src/streamlit_app_clean.py", 
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Rubby Ducky web interface stopped.")
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        print("Make sure streamlit is installed: pip install streamlit")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Rubby Ducky - Rubber Duck Debugging Assistant with Bug Detection")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a code file for bugs')
    analyze_parser.add_argument('file', help='Code file to analyze')
    analyze_parser.add_argument('--model', default='microsoft/codebert-base', help='Model to use for AI analysis')
    analyze_parser.add_argument('--language', '--lang', default='python', 
                               choices=['python', 'javascript', 'java', 'cpp', 'csharp', 'go', 'rust'],
                               help='Programming language: python, javascript, java, cpp (C++), csharp (C#), go, rust')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model on coding datasets')
    train_parser.add_argument('--output', default='./models/coding-assistant', help='Output directory for trained model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--languages', nargs='+', default=['python'],
                             choices=['python', 'javascript', 'java', 'cpp', 'csharp', 'go', 'rust'],
                             help='Programming languages: python, javascript, java, cpp (C++), csharp (C#), go, rust')
    train_parser.add_argument('--train-ml', action='store_true', 
                             help='Train ML bug detection models (requires datasets)')
    train_parser.add_argument('--load-datasets', action='store_true',
                             help='Load and display available bug detection datasets')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=8501, help='Port for web interface')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo analysis on test code')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        result = run_analysis(args.file, args.model, args.language)
        print(f"Analysis for {args.language} file: {args.file}")
        print("=" * 60)
        print(result)
    
    elif args.command == 'train':
        result = run_training(args.output, args.epochs, args.languages, 
                            getattr(args, 'train_ml', False), 
                            getattr(args, 'load_datasets', False))
        print(f"Training completed for languages: {', '.join(args.languages)}")
        print(result)
    
    elif args.command == 'web':
        run_web_interface(args.port)
    
    elif args.command == 'demo':
        # Create a demo file with intentional bugs
        demo_code = '''def calculate_average(numbers):
    if len(numbers) = 0:  # Bug: assignment instead of comparison
        return 0
    
    total = 0
    for num in numbers:
        total += num
    
    try:
        average = total / len(numbers)
        print("Average is: " + str(average))  # Could use f-string
        return average
    except:  # Bug: too broad exception
        print("Error calculating average")
        return None

def process_data(data):
    results = []
    for item in data:
        if item > 0:
            results.append(item * 2)
    return results
'''
        
        # Save demo code to temporary file
        demo_file = 'demo_code_buggy.py'
        with open(demo_file, 'w') as f:
            f.write(demo_code)
        
        print("🦆 Running Rubby Ducky demo analysis...")
        print("This demo shows how Rubby Ducky detects bugs in Python code.\n")
        run_analysis(demo_file, language='python')
        
        # Clean up
        os.remove(demo_file)
        print(f"\n✨ Demo completed! Try 'python main.py web' to launch the web interface.")
        print("Or analyze your own files with: python main.py analyze <file> --language <lang>")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
