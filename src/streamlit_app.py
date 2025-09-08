"""
Streamlit web interface for the AI Coding Assistant.
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import CodeDataProcessor
import json


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="🤖 AI Coding Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🤖 AI Coding Assistant")
    st.markdown("""
    **Detect bugs and get debugging help for your Python code using AI analysis.**
    
    This tool helps you find syntax errors, common bugs, and code quality issues.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Options")
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Static Analysis (Fast)", "AI Analysis (Requires Model)"],
            help="Static analysis works without GPU, AI analysis provides deeper insights"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        - **Static Analysis**: Fast bug detection using Python AST
        - **AI Analysis**: Advanced analysis using Llama models
        - **Real-time**: Instant feedback as you type
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Features")
        st.markdown("""
        ✅ Syntax error detection  
        ✅ Common bug patterns  
        ✅ Code quality suggestions  
        ✅ Best practice recommendations  
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Code Input")
        
        # Code input area
        code_input = st.text_area(
            "Enter your Python code:",
            height=400,
            placeholder="""def example_function(x, y):
    if x = 5:  # Bug: assignment instead of comparison
        result = x + y
        print("Result: " + str(result))  # Could use f-string
    return result

try:
    value = example_function(3, 4)
except:  # Bug: too broad exception
    print("Error occurred")""",
            help="Paste your Python code here for analysis"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Code", type="primary")
        
        # Example buttons
        st.markdown("**Quick Examples:**")
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            if st.button("📋 Load Buggy Code"):
                st.session_state['code_example'] = """def calculate_average(numbers):
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
        return None"""
        
        with col_ex2:
            if st.button("✅ Load Clean Code"):
                st.session_state['code_example'] = """def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    
    total = sum(numbers)
    average = total / len(numbers)
    print(f"Average is: {average}")
    return average

def main():
    data = [1, 2, 3, 4, 5]
    result = calculate_average(data)
    print(f"Result: {result}")

if __name__ == "__main__":
    main()"""
        
        # Use example code if set
        if 'code_example' in st.session_state:
            code_input = st.session_state['code_example']
            # Clear the example from session state
            del st.session_state['code_example']
            st.rerun()
    
    with col2:
        st.header("📊 Analysis Results")
        
        if analyze_button and code_input.strip():
            # Perform analysis
            with st.spinner("Analyzing code..."):
                processor = CodeDataProcessor()
                
                # Static analysis
                syntax_errors = processor.detect_syntax_errors(code_input)
                common_bugs = processor.detect_common_bugs(code_input)
                
                total_issues = len(syntax_errors) + len(common_bugs)
                
                # Display summary
                if total_issues == 0:
                    st.success("🎉 No obvious issues detected! Your code looks good.")
                else:
                    st.warning(f"⚠️ Found {total_issues} potential issues")
                
                # Display syntax errors
                if syntax_errors:
                    st.subheader("🚨 Syntax Errors (High Priority)")
                    for i, error in enumerate(syntax_errors, 1):
                        with st.expander(f"Error {i}: {error['type']}", expanded=True):
                            st.error(f"**Line {error.get('line', '?')}**: {error['message']}")
                            st.markdown("**Severity**: High 🔴")
                            st.markdown("**Action**: Fix this syntax error before running the code.")
                
                # Display common bugs
                if common_bugs:
                    st.subheader("⚠️ Code Quality Issues")
                    for i, bug in enumerate(common_bugs, 1):
                        severity = bug.get('severity', 'unknown')
                        severity_color = {
                            'high': '🔴',
                            'medium': '🟡', 
                            'low': '🟢'
                        }.get(severity, '⚪')
                        
                        with st.expander(f"Issue {i}: {bug['type']}", expanded=True):
                            if severity == 'high':
                                st.error(f"**Line {bug.get('line', '?')}**: {bug['message']}")
                            else:
                                st.warning(f"**Line {bug.get('line', '?')}**: {bug['message']}")
                            st.markdown(f"**Severity**: {severity.title()} {severity_color}")
                            
                            # Provide specific suggestions
                            if bug['type'] == 'AssignmentInCondition':
                                st.markdown("**Fix**: Change `=` to `==` for comparison")
                                st.code("# Instead of: if x = 5:\n# Use: if x == 5:")
                            elif bug['type'] == 'BroadException':
                                st.markdown("**Fix**: Use specific exception types")
                                st.code("# Instead of: except:\n# Use: except ValueError:\n#   or: except (ValueError, TypeError):")
                            elif bug['type'] == 'StringConcatenation':
                                st.markdown("**Fix**: Use f-strings for better performance")
                                st.code('# Instead of: "Result: " + str(value)\n# Use: f"Result: {value}"')
                            elif bug['type'] == 'TypeMismatchConcatenation':
                                st.markdown("**Fix**: Convert number to string or use f-string")
                                st.code('# Instead of: "hello" + 5\n# Use: "hello" + str(5)\n# Or: f"hello{5}"')
                            elif bug['type'] == 'DivisionByZero':
                                st.markdown("**Fix**: Add zero check before division")
                                st.code("# Instead of: result = x / 0\n# Use: if denominator != 0:\n#    result = x / denominator")
                            elif bug['type'] == 'PossibleUndefinedVariable':
                                st.markdown("**Fix**: Define variable before use")
                                st.code("# Make sure to define the variable first:\n# variable_name = some_value\n# print(variable_name)")
                
                # AI Analysis section
                if analysis_type == "AI Analysis (Requires Model)":
                    st.subheader("🤖 AI Analysis")
                    st.info("""
                    **AI Analysis requires model loading.**
                    
                    To enable AI analysis:
                    1. Install additional dependencies: `pip install torch transformers`
                    2. Load a model (requires GPU with 8GB+ VRAM)
                    3. Use the full version of the application
                    
                    For now, static analysis provides comprehensive bug detection.
                    """)
                
                # Code metrics
                st.subheader("📈 Code Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric("Lines of Code", len(code_input.splitlines()))
                
                with col_m2:
                    st.metric("Issues Found", total_issues)
                
                with col_m3:
                    severity_score = 0
                    for error in syntax_errors:
                        severity_score += 3  # High severity
                    for bug in common_bugs:
                        severity_weights = {'high': 3, 'medium': 2, 'low': 1}
                        severity_score += severity_weights.get(bug.get('severity', 'low'), 1)
                    
                    max_possible = total_issues * 3 if total_issues > 0 else 1
                    severity_percentage = (severity_score / max_possible) * 100
                    
                    st.metric(
                        "Severity Score", 
                        f"{severity_percentage:.0f}%",
                        help="Higher percentage indicates more critical issues"
                    )
        
        elif analyze_button and not code_input.strip():
            st.warning("Please enter some Python code to analyze.")
        else:
            st.info("👆 Enter your Python code and click 'Analyze Code' to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🤖 <strong>AI Coding Assistant</strong> - Powered by Static Analysis & Llama Models</p>
        <p>Built with ❤️ using Streamlit | <a href="https://github.com/Daniel011503/AI-Coding-Assist-" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
