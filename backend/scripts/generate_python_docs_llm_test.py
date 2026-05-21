"""
Generate Dummy Data: Python Documentation
Creates realistic technical documentation for testing LLM sumarisation in the pipeline.
"""

import json
import random
from datetime import datetime

# Python documentation examples (realistic but synthetic)
PYTHON_DOCS = [
    {
        "topic": "String Methods",
        "content": """
The str.split() method splits a string into a list of substrings based on a delimiter. 
By default, it splits on whitespace. You can specify a custom delimiter as the first argument.
The maxsplit parameter controls the maximum number of splits. Returns a list of strings.
Example: 'hello world'.split() returns ['hello', 'world'].
"""
    },
    {
        "topic": "List Comprehensions",
        "content": """
List comprehensions provide a concise way to create lists in Python. The syntax is 
[expression for item in iterable if condition]. This is more readable than using 
traditional for loops. Example: squares = [x**2 for x in range(10)] creates a list 
of squares from 0 to 81. You can also add conditional filtering.
"""
    },
    {
        "topic": "Dictionary Methods",
        "content": """
The dict.get() method retrieves a value from a dictionary by key. Unlike bracket notation,
it doesn't raise KeyError if the key doesn't exist. Instead, it returns None or a default
value you specify as the second argument. Example: my_dict.get('key', 'default') returns
'default' if 'key' is not found. Very useful for safely accessing dictionary values.
"""
    },
    {
        "topic": "File Handling",
        "content": """
Python uses the open() function to handle files. The mode parameter specifies how to open
the file: 'r' for reading, 'w' for writing, 'a' for appending. Always use the with statement
to ensure files are properly closed. Example: with open('file.txt', 'r') as f: content = f.read().
This automatically closes the file even if an error occurs.
"""
    },
    {
        "topic": "Exception Handling",
        "content": """
The try-except block catches and handles exceptions in Python. Put risky code in the try block
and error handling in the except block. You can catch specific exceptions or use a general
Exception class. The finally block always executes, useful for cleanup. Example: try/except
ValueError catches only value-related errors. Use raise to re-raise exceptions.
"""
    },
    {
        "topic": "Lambda Functions",
        "content": """
Lambda functions are small anonymous functions defined with the lambda keyword. Syntax is
lambda arguments: expression. They can have any number of arguments but only one expression.
Commonly used with map(), filter(), and sorted() functions. Example: lambda x: x * 2 doubles
a value. Useful for simple operations but less readable than regular functions for complex logic.
"""
    },
    {
        "topic": "Class Inheritance",
        "content": """
Python supports class inheritance using parentheses after the class name. The child class
inherits all methods and attributes from the parent class. Use super() to call parent class
methods. You can override parent methods by defining them again in the child class.
Example: class Child(Parent) creates a child class. Multiple inheritance is also supported.
"""
    },
    {
        "topic": "Decorators",
        "content": """
Decorators are functions that modify the behavior of other functions. Use the @ symbol to
apply them. They wrap the original function and can execute code before/after it runs.
Common use cases include logging, authentication, and caching. Example: @property makes
a method accessible like an attribute. You can also create custom decorators using
nested functions or classes.
"""
    },
    {
        "topic": "Generators",
        "content": """
Generators are functions that use yield instead of return. They produce values lazily,
one at a time, which is memory efficient for large datasets. Each yield pauses the function
and returns a value. Calling next() resumes execution. Example: a generator for Fibonacci
numbers yields each number without storing the entire sequence in memory. Use generator
expressions for simple cases: (x**2 for x in range(10)).
"""
    },
    {
        "topic": "Context Managers",
        "content": """
Context managers handle resource setup and cleanup automatically using the with statement.
They implement __enter__ and __exit__ methods. The contextlib module provides utilities
for creating custom context managers. Example: with open() handles file closing automatically.
You can create custom managers using @contextmanager decorator. Useful for database connections,
locks, and temporary state changes.
"""
    },
    {
        "topic": "Virtual Environments",
        "content": """
Virtual environments isolate Python dependencies per project. Create one using python -m venv
followed by the environment name. Activate with source venv/bin/activate on Unix or
venv\\Scripts\\activate on Windows. Install packages with pip while activated - they only
affect this environment. Deactivate with deactivate command. Essential for managing
different project requirements without conflicts.
"""
    },
    {
        "topic": "Package Installation",
        "content": """
Use pip to install Python packages from PyPI. Basic syntax: pip install package_name.
Install specific versions with pip install package==1.0.0. Use requirements.txt to list
all dependencies: pip install -r requirements.txt. Upgrade packages with --upgrade flag.
Uninstall with pip uninstall package_name. Check installed packages with pip list or pip freeze.
"""
    },
    {
        "topic": "String Formatting",
        "content": """
Python offers multiple ways to format strings. F-strings (Python 3.6+) are most modern:
f'Hello {name}'. The format() method: 'Hello {}'.format(name). Old-style % formatting:
'Hello %s' % name. F-strings support expressions: f'{x + y}' and formatting specs:
f'{value:.2f}' for two decimal places. Most readable and fastest option for string interpolation.
"""
    },
    {
        "topic": "Regular Expressions",
        "content": """
The re module provides regex support in Python. Common functions: re.search() finds first
match, re.findall() returns all matches, re.sub() replaces matches. Use raw strings (r'pattern')
for patterns. Special characters: . matches any character, * means zero or more, + means one
or more, ? makes optional. Groups capture subpatterns with parentheses. Compile patterns
with re.compile() for reuse.
"""
    },
    {
        "topic": "JSON Handling",
        "content": """
The json module handles JSON data in Python. Use json.loads() to parse JSON strings into
Python objects and json.dumps() to convert Python objects to JSON strings. For files, use
json.load() and json.dump(). The indent parameter in dumps() pretty-prints output. Set
ensure_ascii=False for Unicode characters. Custom objects need special encoding with
cls parameter or default function.
"""
    },
    {
        "topic": "Threading Basics",
        "content": """
The threading module enables concurrent execution in Python. Create threads with
threading.Thread(target=function). Start with .start() method and wait with .join().
Thread-safe operations use threading.Lock() to prevent race conditions. Note: Python's GIL
limits true parallelism for CPU-bound tasks. Use multiprocessing for CPU-intensive work.
Threading works well for I/O-bound operations like network requests.
"""
    },
    {
        "topic": "Type Hints",
        "content": """
Type hints (PEP 484) add optional type information to Python code. Syntax: def func(x: int) -> str.
Use typing module for complex types: List[int], Dict[str, Any], Optional[str]. Type hints
don't enforce types at runtime but help with IDE autocomplete and static analysis tools like
mypy. Generic types use TypeVar. Useful for large codebases and API documentation.
"""
    },
    {
        "topic": "Async/Await",
        "content": """
Async/await enables asynchronous programming in Python. Define async functions with
async def and call them with await. Use asyncio.run() to execute async code. Async functions
return coroutines, not results directly. Great for I/O-bound operations like HTTP requests.
asyncio.gather() runs multiple coroutines concurrently. Not suitable for CPU-bound tasks.
Requires async-compatible libraries.
"""
    },
    {
        "topic": "Dataclasses",
        "content": """
Dataclasses (Python 3.7+) reduce boilerplate for data-holding classes. Use @dataclass
decorator to auto-generate __init__, __repr__, and __eq__ methods. Field defaults and
types are defined as class attributes. Set frozen=True for immutability. The field()
function provides advanced options like default_factory for mutable defaults. More
concise than traditional classes for simple data structures.
"""
    },
    {
        "topic": "Pathlib",
        "content": """
The pathlib module provides object-oriented filesystem paths. Path objects have methods
for common operations: .exists(), .is_file(), .is_dir(), .mkdir(). Use / operator to join
paths: Path('folder') / 'file.txt'. Read/write with .read_text() and .write_text().
Glob patterns with .glob('*.txt'). Cross-platform compatible - handles Windows/Unix path
differences automatically. More intuitive than os.path for modern code.
"""
    }
]


def generate_python_docs_file(output_path: str = "python_docs_dataset.txt", num_repeats: int = 2):
    """
    Generate a text file with Python documentation content.
    
    Args:
        output_path: Where to save the file
        num_repeats: How many times to repeat the dataset (for more volume)
    """
    all_content = []
    
    for _ in range(num_repeats):
        for doc in PYTHON_DOCS:
            # Add topic header and content
            section = f"=== {doc['topic']} ===\n\n{doc['content'].strip()}\n\n"
            all_content.append(section)
    
    # Shuffle for variety
    random.shuffle(all_content)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("PYTHON PROGRAMMING DOCUMENTATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(''.join(all_content))
    
    print(f"Generated {output_path}")
    print(f"  - {len(all_content)} sections")
    print(f"  - {sum(len(c) for c in all_content)} total characters")
    print(f"  - Estimated {len(all_content) * 2} chunks (500 char chunks)")


def generate_json_format(output_path: str = "python_docs_dataset.json"):
    """Generate JSON format for direct database insertion."""
    data = []
    
    for doc in PYTHON_DOCS:
        data.append({
            "topic": doc["topic"],
            "content": doc["content"].strip(),
            "category": "python_documentation",
            "created_at": datetime.now().isoformat()
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {output_path}")


if __name__ == "__main__":
    # Generate both formats
    generate_python_docs_file("python_docs_llm_test_dataset.txt", num_repeats=3)
    generate_json_format("python_docs_llm_test_dataset.json")
    
    print("\nUsage:")
    print("1. Upload python_docs_llm_test_dataset.txt via /upload-file-for-pipeline")
    print("2. Run pipeline with your choice of chunking/embedding")
    print("3. Test LLM summary with queries like:")
    print("   - 'How do I split strings in Python?'")
    print("   - 'Explain list comprehensions'")
    print("   - 'What are the best practices for file handling?'")
    print("   - 'How do virtual environments work?'")