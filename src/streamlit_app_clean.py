"""
Streamlit web interface for Rubby Ducky - The Rubber Duck Debugging Assistant.
"""

import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.multi_language_processor import MultiLanguageCodeProcessor
import json


def get_example_code(language: str, buggy: bool = True):
    """Get example code for the specified language."""
    # Map display names to internal language codes
    language_map = {
        'Python': 'python',
        'JavaScript': 'javascript', 
        'Java': 'java',
        'C++': 'cpp',
        'C#': 'csharp',
        'Go': 'go',
        'Rust': 'rust'
    }
    
    # Convert display name to internal code
    internal_lang = language_map.get(language, language.lower())
    
    file_extensions = {
        'python': 'py',
        'javascript': 'js', 
        'java': 'java',
        'cpp': 'cpp',
        'csharp': 'cs',
        'go': 'go',
        'rust': 'rs'
    }
    
    if internal_lang not in file_extensions:
        return f"# Language '{language}' not supported\nprint('Hello, World!')"
    
    ext = file_extensions[internal_lang]
    folder = 'buggy' if buggy else 'clean'
    filename = f"test_samples/{folder}/{internal_lang}_{folder}.{ext}"
    
    try:
        # Try to read from test files first
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to simple examples if test files don't exist
        examples = {
            'python': {
                'buggy': 'def calculate_average(numbers):\n    if len(numbers) = 0:  # Bug: assignment instead of comparison\n        return 0\n    return sum(numbers) / len(numbers)',
                'clean': 'def calculate_average(numbers):\n    if len(numbers) == 0:\n        return 0\n    return sum(numbers) / len(numbers)'
            },
            'javascript': {
                'buggy': 'function calculateAverage(numbers) {\n    if (numbers.length = 0) {  // Bug: assignment instead of comparison\n        return 0;\n    }\n    return numbers.reduce((a, b) => a + b, 0) / numbers.length;\n}',
                'clean': 'function calculateAverage(numbers) {\n    if (numbers.length === 0) {\n        return 0;\n    }\n    return numbers.reduce((a, b) => a + b, 0) / numbers.length;\n}'
            }
        }
        
        # Get the appropriate example or default to Python
        lang_examples = examples.get(language, examples['python'])
        return lang_examples['buggy'] if buggy else lang_examples['clean']


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="🦆 Rubby Ducky - Rubber Duck Debugging Assistant",
        page_icon="🦆",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🦆 Rubby Ducky")
    st.markdown("### *Your Rubber Duck Debugging Assistant*")
    
    st.markdown("""
    **Your Rubber Duck Debugging Assistant - AI-powered bug detection and code analysis.**
    
    🦆 **Rubber Duck Debugging**: Explain your code to Rubby and discover issues through conversation  
    🧠 **Neural Network Analysis**: Uses trained CodeBERT models for intelligent bug detection  
    📊 **Rule-Based Static Analysis**: 40+ comprehensive rules across all languages  
    🌐 **Multi-Language Support**: Full AI analysis for Python, JavaScript, Java, C++, C#, Go, and Rust  
    ⚡ **Real-Time Analysis**: Instant feedback with confidence scores and actionable suggestions
    """)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Languages Supported", "7", "Full AI Analysis")
    with col2:
        st.metric("🧮 Analysis Rules", "40+", "Database-Driven")
    with col3:
        st.metric("🔬 ML Models", "CodeBERT", "Neural Networks")
    with col4:
        st.metric("📊 Bug Categories", "6+", "Security, Performance")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Analysis Options")
        
        # Language selection
        language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust"],
            index=0,
            key="language_selector",
            help="All languages now support full AI analysis!"
        )
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["🤖 AI Analysis (Neural Networks + Rules)", "⚡ Static Analysis (Rules Only)"],
            key="analysis_type_selector",
            help="AI analysis combines neural networks with rule-based detection for maximum accuracy"
        )
        
        st.markdown("---")
        
        # Quick examples section
        st.subheader("📝 Quick Examples")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Load Clean Code", key="load_clean"):
                st.session_state.code_input = get_example_code(language, buggy=False)
                st.success("✅ Clean code loaded!")
                st.rerun()
                
        with col2:
            if st.button("🐛 Load Buggy Code", key="load_buggy"):
                st.session_state.code_input = get_example_code(language, buggy=True)
                st.success("🐛 Buggy code loaded!")
                st.rerun()
    
    # Main content area
    st.header("🔍 Code Analysis")
    
    # Code input
    code_input = st.text_area(
        "Enter your code here:",
        height=300,
        value=st.session_state.get('code_input', ''),
        key="code_area",
        help="Paste your code or use the Quick Examples buttons in the sidebar"
    )
    
    # Analysis button
    if st.button("🦆 Analyze with Rubby Ducky", type="primary", key="analyze_button"):
        if code_input.strip():
            try:
                with st.spinner("🦆 Rubby is analyzing your code..."):
                    # Map display name to internal language code
                    language_map = {
                        'Python': 'python',
                        'JavaScript': 'javascript', 
                        'Java': 'java',
                        'C++': 'cpp',
                        'C#': 'csharp',
                        'Go': 'go',
                        'Rust': 'rust'
                    }
                    internal_lang = language_map.get(language, language.lower())
                    
                    # Initialize processor
                    use_ml = "AI Analysis" in analysis_type
                    processor = MultiLanguageCodeProcessor(language=internal_lang, use_ml_model=use_ml)
                    
                    # Get ML analysis if enabled
                    if use_ml:
                        ml_result = processor.predict_bugs_ml(code_input)
                        
                        # Get rule-based analysis first to inform AI classification
                        syntax_errors = processor.detect_syntax_errors(code_input, internal_lang)
                        common_bugs = processor.detect_common_bugs(code_input, internal_lang)
                        
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
                        
                        # For display: show bug probability if buggy, clean confidence if clean
                        if final_prediction == 'buggy':
                            confidence = final_confidence  # Bug probability
                            is_buggy = True
                        else:
                            confidence = 1.0 - final_confidence  # Convert clean confidence to bug probability for display
                            is_buggy = False
                    else:
                        confidence = 0.0
                        is_buggy = False
                        # Get rule-based analysis
                        syntax_errors = processor.detect_syntax_errors(code_input, internal_lang)
                        common_bugs = processor.detect_common_bugs(code_input, internal_lang)
                
                # Combine all issues for display
                all_issues = syntax_errors + common_bugs
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if use_ml and 'final_prediction' in locals():
                        if final_prediction == 'clean':
                            st.metric("🤖 AI Analysis", f"Clean ({final_confidence:.1%})", delta="🟢")
                        else:
                            st.metric("🤖 AI Analysis", f"Buggy ({final_confidence:.1%})", delta="🔴")
                    else:
                        confidence_color = "🔴" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🟢"
                        st.metric("🤖 Bug Probability", f"{confidence:.1%}", delta=confidence_color)
                with col2:
                    st.metric("🔍 Issues Found", len(all_issues))
                with col3:
                    severity_high = sum(1 for issue in all_issues if issue.get('severity') == 'high')
                    st.metric("🚨 High Severity", severity_high)
                
                # Issues breakdown
                if all_issues:
                    st.subheader("🐛 Issues Found")
                    
                    # Group by severity
                    high_issues = [i for i in all_issues if i.get('severity') == 'high']
                    medium_issues = [i for i in all_issues if i.get('severity') == 'medium']
                    low_issues = [i for i in all_issues if i.get('severity') == 'low']
                    
                    if high_issues:
                        st.error(f"🚨 High Severity Issues ({len(high_issues)})")
                        for issue in high_issues[:5]:  # Show top 5
                            st.write(f"**Line {issue['line']}**: {issue['message']}")
                            if 'suggestion' in issue:
                                st.write(f"💡 *{issue['suggestion']}*")
                    
                    if medium_issues:
                        st.warning(f"⚠️ Medium Severity Issues ({len(medium_issues)})")
                        for issue in medium_issues[:3]:  # Show top 3
                            st.write(f"**Line {issue['line']}**: {issue['message']}")
                            if 'suggestion' in issue:
                                st.write(f"💡 *{issue['suggestion']}*")
                    
                    if low_issues:
                        st.info(f"ℹ️ Low Severity Issues ({len(low_issues)})")
                        for issue in low_issues[:3]:  # Show top 3
                            st.write(f"**Line {issue['line']}**: {issue['message']}")
                else:
                    st.success("🎉 No issues found! Your code looks great!")
                    st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error analyzing code: {str(e)}")
        else:
            st.warning("⚠️ Please enter some code to analyze!")
    
    # Update session state
    if code_input != st.session_state.get('code_input', ''):
        st.session_state.code_input = code_input
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🦆 <strong>Rubby Ducky</strong> | Your Rubber Duck Debugging Assistant powered by AI</p>
        <p>🌟 AI-enhanced analysis for all 7 programming languages with 40+ detection rules!</p>
        <p>Built with ❤️ using Streamlit | <a href="https://github.com/Daniel011503/Rubby-Ducky" target="_blank">View on GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
