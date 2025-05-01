import json
import re
import argparse
import sys
from collections import defaultdict

try:
    from stdlib_list import stdlib_list
except ImportError:
    stdlib_list = None

def is_builtin_module(name):
    """Check if a module is part of the Python standard library."""
    # Use sys.stdlib_module_names if available (Python 3.10+)
    if hasattr(sys, 'stdlib_module_names'):
        return name in sys.stdlib_module_names
    # Fallback to stdlib-list if installed
    elif stdlib_list is not None:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        stdlib_modules = stdlib_list(version)
        return name in stdlib_modules
    else:
        # Fallback to a basic hard-coded list for demonstration
        # Consider extending this or installing stdlib-list for better accuracy
        builtins = {
            'os', 'sys', 're', 'math', 'json', 'time', 'datetime',
            'collections', 'itertools', 'functools', 'random', 'typing',
            'io', 'pathlib', 'abc', 'ast', 'base64', 'bisect', 'calendar',
            'copy', 'csv', 'ctypes', 'dataclasses', 'decimal', 'difflib',
            'enum', 'fractions', 'hashlib', 'heapq', 'hmac', 'inspect',
            'logging', 'multiprocessing', 'numbers', 'operator', 'queue',
            'secrets', 'signal', 'sqlite3', 'statistics', 'string', 'struct',
            'threading', 'timeit', 'types', 'uuid', 'warnings', 'sysconfig',
            'importlib', 'argparse'
        }
        return name.lower() in {m.lower() for m in builtins}

def extract_imports(code_string):
    """Extract import statements from Python code string."""
    if not code_string or not isinstance(code_string, str):
        return set()
        
    standard_imports = re.findall(r'(?:^|\n)\s*import\s+([\w\.]+)(?:\s+as\s+[\w\.]+)?', code_string)
    from_imports = re.findall(r'(?:^|\n)\s*from\s+([\w\.]+)\s+import', code_string)
    
    all_imports = set()
    for imp in standard_imports + from_imports:
        top_package = imp.split('.')[0]
        if not is_builtin_module(top_package) and is_likely_package(top_package):
            all_imports.add(top_package)
    
    return all_imports

def is_likely_package(name):
    """Simple heuristic to filter out non-package imports."""
    # Skip single-letter names (likely variables)
    if len(name) == 1:
        return False
        
    # Skip names with uppercase first letter (likely classes)
    if name[0].isupper():
        return False
    
    # Common Python package naming patterns
    if not re.match(r'^[a-z][a-z0-9_]*$', name):
        return False
        
    # Classes, functions, and other common non-packages
    common_non_packages = {
        'array', 'deque', 'defaultdict', 'namedtuple', 'counter', 'ordereddict', 'dataclass',
        'arange', 'ndarray'
    }
    
    if name.lower() in common_non_packages:
        return False
    
    # Well-known third-party packages
    known_packages = {
        'numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'tensorflow', 'torch', 
        'keras', 'sympy', 'nltk', 'networkx', 'PIL', 'cv2', 'requests', 'bs4',
        'selenium', 'flask', 'django', 'plotly', 'seaborn', 'pytest', 'numba',
        'xgboost', 'transformers', 'sqlalchemy', 'pydantic', 'fastapi', 'pyspark',
        'statsmodels', 'z3', 'mpmath', 'gmpy2', 'pulp', 'ortools', 'crypto', 'cryptography',
        'yaml', 'dateutil'
    }
    
    if name.lower() in known_packages:
        return True
    
    # Default to accepting things that look like packages
    return True

# PyPI package name mapping for common discrepancies
PYPI_PACKAGE_MAPPING = {
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'bs4': 'beautifulsoup4',
    'cv2': 'opencv-python',
    'yaml': 'pyyaml',
    'dateutil': 'python-dateutil'
}

def process_jsonl(jsonl_path):
    """Process JSONL file and extract imports from code snippets."""
    all_packages = defaultdict(int)
    records_processed = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                records_processed += 1
                
                # Look for a 'code' field directly in the data
                if 'code' in data and data['code']:
                    imports = extract_imports(data['code'])
                    for imp in imports:
                        all_packages[imp] += 1
                    continue
                    
                # Look for code in referenced code snippets
                for key, value in data.items():
                    # If the key contains 'code' and value is a string, it might be code
                    if 'code' in key.lower() and isinstance(value, str):
                        imports = extract_imports(value)
                        for imp in imports:
                            all_packages[imp] += 1
                            
                # If we couldn't find dedicated code fields, try to extract from prompt as fallback
                # This can happen in some datasets where code is embedded in the prompt
                if 'prompt' in data and data['prompt'] and len(all_packages) == 0:
                    # Look for code blocks in the prompt (code enclosed in triple backticks)
                    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', data['prompt'], re.DOTALL)
                    for block in code_blocks:
                        imports = extract_imports(block)
                        for imp in imports:
                            all_packages[imp] += 1
                    
                    # If no code blocks found, try checking if the whole prompt is code
                    if not code_blocks and '=' in data['prompt'] and 'def ' in data['prompt']:
                        imports = extract_imports(data['prompt'])
                        for imp in imports:
                            all_packages[imp] += 1
                            
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON. Skipping.")
                continue
    
    print(f"Processed {records_processed} records from the JSONL file.")
    return all_packages

def create_conda_env_file(packages, output_path):
    """Create conda environment YAML file with required packages."""
    yaml_content = "name: code_execution_env\nchannels:\n  - defaults\n  - conda-forge\ndependencies:\n  - python=3.9\n"
    
    # Add conda packages with name mapping
    for package in sorted(packages.keys()):
        pkg = PYPI_PACKAGE_MAPPING.get(package, package)
        yaml_content += f"  - {pkg}\n"
    
    # Add pip section for packages that might need it
    yaml_content += "  - pip\n  - pip:\n"
    for package in sorted(packages.keys()):
        pkg = PYPI_PACKAGE_MAPPING.get(package, package)
        yaml_content += f"    - {pkg}\n"
    
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Generated conda environment file at {output_path}")
    print(f"Found {len(packages)} packages: {', '.join(sorted(packages.keys()))}")

def main():
    parser = argparse.ArgumentParser(description='Extract Python package dependencies from JSONL file with code snippets')
    parser.add_argument('jsonl_path', help='Path to the JSONL file')
    parser.add_argument('--output', default='codeIO_dependencies.yml', help='Output path for conda environment file')
    parser.add_argument('--min-occurrences', type=int, default=2, help='Minimum number of occurrences required to include a package')
    
    args = parser.parse_args()
    
    print(f"Processing JSONL file: {args.jsonl_path}")
    required_packages = process_jsonl(args.jsonl_path)
    
    # Filter packages by frequency
    significant_packages = {pkg: count for pkg, count in required_packages.items() if count >= args.min_occurrences}
    
    create_conda_env_file(significant_packages, args.output)

if __name__ == "__main__":
    main()