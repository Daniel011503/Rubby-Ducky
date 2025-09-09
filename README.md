# 🤖 AI Coding Assistant

An intelligent coding assistant that detects bugs and provides debugging suggestions using Llama models from Hugging Face.

## ✨ Features

- 🔍 **Bug Detection**: Identify syntax errors and common coding issues
- 🤖 **AI-Powered Analysis**: Leverage Llama models for advanced code review
- 🛠️ **Debug Assistance**: Get step-by-step debugging help
- 🌐 **Web Interface**: User-friendly Gradio interface
- 📊 **Training Pipeline**: Fine-tune models on coding datasets
- 🎯 **Static Analysis**: Fast analysis without requiring GPU

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Daniel011503/AI-Coding-Assist-.git
cd AI-Coding-Assist-

# Create and activate virtual environment
python -m venv llama-env
# On Windows:
llama-env\Scripts\activate
# On Linux/Mac:
source llama-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Usage Options

#### Option A: Web Interface (Recommended)

```bash
python main.py web
```

Then open http://localhost:7860 in your browser.

#### Option B: Command Line Analysis

```bash
# Analyze a specific Python file
python main.py analyze your_code.py

# Run demo analysis
python main.py demo
```

#### Option C: Direct Python Usage

```python
from src.inference import CodingAssistant

# Initialize assistant
assistant = CodingAssistant()

# For static analysis (no GPU required)
from src.data_processing import CodeDataProcessor
processor = CodeDataProcessor()
issues = processor.detect_syntax_errors(code) + processor.detect_common_bugs(code)

# For AI analysis (requires GPU and model loading)
assistant.load_model()
result = assistant.analyze_code(code)
```

## 🧠 How It Works

### 1. Static Analysis

- **Syntax Error Detection**: Uses Python AST to find syntax issues
- **Pattern Matching**: Identifies common bugs like assignment in conditions
- **Code Quality**: Checks for best practices violations

### 2. AI Analysis

- **Model**: Uses CodeLlama-7b-Python specialized for code understanding
- **Datasets**: Trained on CodeSearchNet, CodeAlpaca, and GitHub code
- **Techniques**: Fine-tuning with LoRA for efficient training

### 3. Bug Categories Detected

| Category           | Examples                           | Severity |
| ------------------ | ---------------------------------- | -------- |
| Syntax Errors      | Missing colons, wrong indentation  | High     |
| Logic Errors       | Assignment vs comparison (= vs ==) | Medium   |
| Exception Handling | Bare except clauses                | Low      |
| Code Style         | String concatenation vs f-strings  | Low      |

## 📁 Project Structure

```
AI-Coding-Assist-/
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Dataset loading and preprocessing
│   ├── training.py           # Model fine-tuning pipeline
│   ├── inference.py          # Bug detection and analysis
│   └── web_interface.py      # Gradio web interface
├── data/                     # Training data (created during use)
├── models/                   # Trained models (created during training)
├── notebooks/                # Jupyter notebooks for experiments
├── tests/                    # Unit tests
├── main.py                   # Main CLI interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Advanced Usage

### Training Your Own Model

```bash
# Train on coding datasets
python main.py train --output ./models/my-coding-assistant --epochs 5
```

### Configuration

The assistant supports various Llama models:

- `codellama/CodeLlama-7b-Python-hf` (recommended)
- `codellama/CodeLlama-13b-Python-hf` (larger, better quality)
- `meta-llama/Llama-2-7b-chat-hf` (general purpose)

### API Usage

```python
from src.inference import CodingAssistant

# Initialize with custom model
assistant = CodingAssistant("codellama/CodeLlama-7b-Python-hf")
assistant.load_model(use_quantization=True)

# Analyze code
result = assistant.analyze_code(your_code)

# Get debugging help
debug_help = assistant.debug_code(buggy_code, error_message)
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ main.py

# Lint code
flake8 src/ main.py

# Type checking
mypy src/ main.py
```

### Adding New Bug Detection Rules

1. Edit `src/data_processing.py`
2. Add pattern to `detect_common_bugs()` method
3. Update training data generation
4. Retrain model if needed

## 📊 Performance

### System Requirements

| Component | Minimum         | Recommended |
| --------- | --------------- | ----------- |
| RAM       | 8GB             | 16GB+       |
| GPU       | None (CPU mode) | 8GB+ VRAM   |
| Storage   | 2GB             | 10GB+       |
| Python    | 3.8+            | 3.9+        |

### Benchmarks

- **Static Analysis**: ~100ms per file
- **AI Analysis**: ~2-5s per file (with GPU)
- **Model Loading**: ~30s (first time)
- **Training**: ~2-4 hours (depends on dataset size)

## 🎯 Examples

### Example 1: Syntax Error Detection

```python
# Input code with bug
def calculate(x, y)  # Missing colon
    return x + y

# Output
{
  "syntax_errors": [
    {
      "type": "SyntaxError",
      "line": 1,
      "message": "invalid syntax",
      "severity": "high"
    }
  ]
}
```

### Example 2: Logic Error Detection

```python
# Input code with bug
if user_age = 18:  # Assignment instead of comparison
    print("Adult")

# Output
{
  "common_bugs": [
    {
      "type": "AssignmentInCondition",
      "line": 1,
      "message": "Possible assignment (=) instead of comparison (==)",
      "severity": "medium"
    }
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing the model infrastructure
- **Meta AI** for the Llama model architecture
- **CodeLlama** team for the specialized coding model
- **Open source community** for datasets and tools

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/Daniel011503/AI-Coding-Assist-/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Daniel011503/AI-Coding-Assist-/discussions)
- 📧 **Email**: [Your contact email]

## 🗺️ Roadmap

- [ ] Support for more programming languages (JavaScript, Java, C++)
- [ ] Integration with VS Code extension
- [ ] Real-time analysis as you type
- [ ] Custom rule creation interface
- [ ] Performance optimization suggestions
- [ ] Security vulnerability detection
- [ ] Code refactoring suggestions

---

⭐ **Star this repository if it helped you!** ⭐
