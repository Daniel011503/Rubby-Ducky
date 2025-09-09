"""
Rule-based static analysis engine.
Loads analysis rules from JSON databases and applies them to code.
"""

import json
import re
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class AnalysisRule:
    """Represents a single analysis rule."""
    id: str
    pattern: str
    message: str
    severity: str
    category: str
    suggestion: str
    compiled_pattern: Optional[re.Pattern] = None
    
    def __post_init__(self):
        """Compile the regex pattern after initialization."""
        try:
            self.compiled_pattern = re.compile(self.pattern, re.MULTILINE)
        except re.error as e:
            print(f"Warning: Invalid regex pattern in rule {self.id}: {e}")
            self.compiled_pattern = None


class RuleEngine:
    """Rule-based static analysis engine."""
    
    def __init__(self, rules_directory: str = None):
        """Initialize the rule engine with rules from the specified directory."""
        if rules_directory is None:
            # Default to data/rules directory relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rules_directory = os.path.join(os.path.dirname(current_dir), "data", "rules")
        
        self.rules_directory = rules_directory
        self.language_rules: Dict[str, Dict[str, List[AnalysisRule]]] = {}
        self.common_rules: Dict[str, List[AnalysisRule]] = {}
        self._load_all_rules()
    
    def _load_all_rules(self):
        """Load all rule files from the rules directory."""
        if not os.path.exists(self.rules_directory):
            print(f"Warning: Rules directory {self.rules_directory} not found")
            return
        
        # Load language-specific rules
        language_files = [
            "python_rules.json",
            "javascript_rules.json", 
            "java_rules.json",
            "cpp_rules.json"
        ]
        
        for file_name in language_files:
            file_path = os.path.join(self.rules_directory, file_name)
            if os.path.exists(file_path):
                self._load_language_rules(file_path)
        
        # Load common rules
        common_file = os.path.join(self.rules_directory, "common_rules.json")
        if os.path.exists(common_file):
            self._load_common_rules(common_file)
    
    def _load_language_rules(self, file_path: str):
        """Load rules for a specific language from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for language, language_data in data.items():
                if language not in self.language_rules:
                    self.language_rules[language] = {
                        'syntax_patterns': [],
                        'bug_patterns': []
                    }
                
                # Load syntax patterns
                if 'syntax_patterns' in language_data:
                    for rule_data in language_data['syntax_patterns']:
                        rule = self._create_rule_from_data(rule_data)
                        if rule:
                            self.language_rules[language]['syntax_patterns'].append(rule)
                
                # Load bug patterns
                if 'bug_patterns' in language_data:
                    for rule_data in language_data['bug_patterns']:
                        rule = self._create_rule_from_data(rule_data)
                        if rule:
                            self.language_rules[language]['bug_patterns'].append(rule)
            
            print(f"✅ Loaded rules from {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"❌ Error loading rules from {file_path}: {e}")
    
    def _load_common_rules(self, file_path: str):
        """Load common rules that apply to all languages."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'common_patterns' in data:
                for category, rules_data in data['common_patterns'].items():
                    if category not in self.common_rules:
                        self.common_rules[category] = []
                    
                    for rule_data in rules_data:
                        rule = self._create_rule_from_data(rule_data)
                        if rule:
                            self.common_rules[category].append(rule)
            
            print(f"✅ Loaded common rules from {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"❌ Error loading common rules from {file_path}: {e}")
    
    def _create_rule_from_data(self, rule_data: Dict[str, Any]) -> Optional[AnalysisRule]:
        """Create an AnalysisRule object from JSON data."""
        try:
            return AnalysisRule(
                id=rule_data['id'],
                pattern=rule_data['pattern'],
                message=rule_data['message'],
                severity=rule_data['severity'],
                category=rule_data['category'],
                suggestion=rule_data['suggestion']
            )
        except KeyError as e:
            print(f"Warning: Missing required field {e} in rule data")
            return None
    
    def analyze_code(self, code: str, language: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze code using the loaded rules.
        
        Args:
            code: The source code to analyze
            language: Programming language of the code
            
        Returns:
            Dictionary with 'syntax_errors' and 'bugs' lists
        """
        results = {
            'syntax_errors': [],
            'bugs': []
        }
        
        # Apply language-specific rules
        if language in self.language_rules:
            lang_rules = self.language_rules[language]
            
            # Check syntax patterns
            for rule in lang_rules.get('syntax_patterns', []):
                violations = self._apply_rule(rule, code)
                results['syntax_errors'].extend(violations)
            
            # Check bug patterns
            for rule in lang_rules.get('bug_patterns', []):
                violations = self._apply_rule(rule, code)
                results['bugs'].extend(violations)
        
        # Apply common rules
        for category, rules in self.common_rules.items():
            for rule in rules:
                violations = self._apply_rule(rule, code)
                results['bugs'].extend(violations)
        
        return results
    
    def _apply_rule(self, rule: AnalysisRule, code: str) -> List[Dict[str, Any]]:
        """Apply a single rule to the code and return violations."""
        violations = []
        
        if not rule.compiled_pattern:
            return violations
        
        lines = code.split('\n')
        
        for line_number, line in enumerate(lines, 1):
            matches = rule.compiled_pattern.finditer(line)
            
            for match in matches:
                violation = {
                    'rule_id': rule.id,
                    'line': line_number,
                    'column': match.start() + 1,
                    'message': rule.message,
                    'severity': rule.severity,
                    'category': rule.category,
                    'suggestion': rule.suggestion,
                    'matched_text': match.group(0),
                    'line_content': line.strip()
                }
                violations.append(violation)
        
        return violations
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded rules."""
        stats = {
            'languages': {},
            'common_categories': {},
            'total_rules': 0
        }
        
        # Count language-specific rules
        for language, rules in self.language_rules.items():
            syntax_count = len(rules.get('syntax_patterns', []))
            bug_count = len(rules.get('bug_patterns', []))
            total = syntax_count + bug_count
            
            stats['languages'][language] = {
                'syntax_rules': syntax_count,
                'bug_rules': bug_count,
                'total': total
            }
            stats['total_rules'] += total
        
        # Count common rules
        for category, rules in self.common_rules.items():
            count = len(rules)
            stats['common_categories'][category] = count
            stats['total_rules'] += count
        
        return stats
    
    def add_custom_rule(self, language: str, rule_type: str, rule_data: Dict[str, Any]):
        """Add a custom rule at runtime."""
        rule = self._create_rule_from_data(rule_data)
        if not rule:
            return False
        
        if language not in self.language_rules:
            self.language_rules[language] = {
                'syntax_patterns': [],
                'bug_patterns': []
            }
        
        if rule_type in ['syntax_patterns', 'bug_patterns']:
            self.language_rules[language][rule_type].append(rule)
            return True
        
        return False


# Example usage and testing
if __name__ == "__main__":
    # Initialize the rule engine
    engine = RuleEngine()
    
    # Print statistics
    stats = engine.get_rule_statistics()
    print("📊 Rule Engine Statistics:")
    print(f"Total rules loaded: {stats['total_rules']}")
    print("\nLanguage-specific rules:")
    for lang, counts in stats['languages'].items():
        print(f"  {lang}: {counts['total']} rules ({counts['syntax_rules']} syntax, {counts['bug_rules']} bugs)")
    
    print("\nCommon rules:")
    for category, count in stats['common_categories'].items():
        print(f"  {category}: {count} rules")
    
    # Test with sample code
    test_code = '''
def test_function():
    if x = 5:  # Bug: assignment in condition
        print("Hello")
    
    eval("dangerous_code")  # Security issue
    '''
    
    print("\n🔍 Testing with sample Python code:")
    results = engine.analyze_code(test_code, 'python')
    
    if results['syntax_errors']:
        print(f"\n🚨 Syntax Errors ({len(results['syntax_errors'])}):")
        for error in results['syntax_errors']:
            print(f"  Line {error['line']}: {error['message']}")
    
    if results['bugs']:
        print(f"\n⚠️ Code Issues ({len(results['bugs'])}):")
        for bug in results['bugs']:
            print(f"  Line {bug['line']} [{bug['severity'].upper()}]: {bug['message']}")
