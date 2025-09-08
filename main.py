"""
Main runner for the AI Coding Assistant.
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import CodingAssistant
from src.training import CodeLlamaTrainer
from src.web_interface import create_interface
from src.data_processing import CodeDataProcessor


def run_analysis(code_file: str, model_name: str = "codellama/CodeLlama-7b-Python-hf"):
    """Run code analysis on a file."""
    print(f"Analyzing code in: {code_file}")
    
    # Read code file
    with open(code_file, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Initialize assistant
    assistant = CodingAssistant(model_name)
    
    try:
        # Load model for AI analysis
        assistant.load_model()
        result = assistant.analyze_code(code)
        
        print("\n" + "="*50)
        print("CODE ANALYSIS REPORT")
        print("="*50)
        
        print(f"\nFile: {code_file}")
        print(f"Lines of code: {len(code.splitlines())}")
        print(f"Severity Score: {result['severity_score']:.2f}")
        
        if result['syntax_errors']:
            print(f"\n🚨 SYNTAX ERRORS ({len(result['syntax_errors'])}):")
            for error in result['syntax_errors']:
                print(f"  - Line {error.get('line', '?')}: {error['message']}")
        
        if result['common_bugs']:
            print(f"\n⚠️ CODE QUALITY ISSUES ({len(result['common_bugs'])}):")
            for bug in result['common_bugs']:
                severity = bug.get('severity', 'unknown').upper()
                print(f"  - [{severity}] Line {bug.get('line', '?')}: {bug['message']}")
        
        if result['suggestions']:
            print(f"\n💡 SUGGESTIONS:")
            for suggestion in result['suggestions']:
                print(f"  - {suggestion}")
        
        if result['ai_analysis']:
            print(f"\n🤖 AI ANALYSIS:")
            print(result['ai_analysis'])
        
        if not result['syntax_errors'] and not result['common_bugs']:
            print("\n✅ No obvious issues detected! Your code looks good.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Falling back to static analysis only...")
        
        # Static analysis fallback
        processor = CodeDataProcessor()
        syntax_errors = processor.detect_syntax_errors(code)
        common_bugs = processor.detect_common_bugs(code)
        
        print("\n" + "="*50)
        print("STATIC CODE ANALYSIS REPORT")
        print("="*50)
        
        if syntax_errors:
            print(f"\n🚨 SYNTAX ERRORS ({len(syntax_errors)}):")
            for error in syntax_errors:
                print(f"  - Line {error.get('line', '?')}: {error['message']}")
        
        if common_bugs:
            print(f"\n⚠️ CODE QUALITY ISSUES ({len(common_bugs)}):")
            for bug in common_bugs:
                severity = bug.get('severity', 'unknown').upper()
                print(f"  - [{severity}] Line {bug.get('line', '?')}: {bug['message']}")
        
        if not syntax_errors and not common_bugs:
            print("\n✅ No obvious issues detected!")


def run_training(output_dir: str, epochs: int = 3):
    """Run model training."""
    print(f"Starting training process...")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    
    trainer = CodeLlamaTrainer()
    trainer.train(
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=2,
        learning_rate=2e-4
    )
    
    print("Training completed!")


def run_web_interface(port: int = 7860):
    """Launch the web interface."""
    print(f"Launching web interface on port {port}")
    print("Access at: http://localhost:7860")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=False
    )


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="AI Coding Assistant - Bug Detection and Debugging")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a Python file for bugs')
    analyze_parser.add_argument('file', help='Python file to analyze')
    analyze_parser.add_argument('--model', default='codellama/CodeLlama-7b-Python-hf', help='Model to use')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model on coding datasets')
    train_parser.add_argument('--output', default='./models/coding-assistant', help='Output directory for trained model')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Launch web interface')
    web_parser.add_argument('--port', type=int, default=7860, help='Port for web interface')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo analysis on test code')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        run_analysis(args.file, args.model)
    
    elif args.command == 'train':
        run_training(args.output, args.epochs)
    
    elif args.command == 'web':
        run_web_interface(args.port)
    
    elif args.command == 'demo':
        # Create a demo file
        demo_code = '''
def calculate_average(numbers):
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
        demo_file = 'demo_code.py'
        with open(demo_file, 'w') as f:
            f.write(demo_code)
        
        print("Running demo analysis...")
        run_analysis(demo_file)
        
        # Clean up
        os.remove(demo_file)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
