import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

def extract_called_functions(function_content, all_functions):
    """Extract and return the content of functions called within a given function."""
    function_calls_pattern = r'\b(\w+)\(\)'
    called_function_names = re.findall(function_calls_pattern, function_content)
    
    called_functions = []
    for name in called_function_names:
        for func in all_functions:
            if func.startswith(f"void {name}("):
                called_functions.append(func)
                break
    return called_functions

def extract_functions(content, pattern):
    """Extract functions from C file content based on a regex pattern."""
    return re.findall(pattern, content, re.DOTALL)

def tokenize_function(function):
    tokens = []
    patterns = {
        'function_declaration': r'\bvoid\s+\w+\s*\([^()]*\)',
        'string_literal': r'\".*?\"',
        'preprocessor_directive': r'#\w+ <.*?>',
        'keyword': r'\b(int|char|float|double|void|if|else|while|for|return|struct)\b',
        'function_call': r'\b\w+\s*\([^()]*\)',
        'number': r'\b\d+\b',
        'identifier': r'\b[a-zA-Z_]\w*\b',
        'operator': r'[+\-*/=<>!&|,.;:{}()\[\]]'
    }

    function = re.sub(r'//.*?\n|/\*.*?\*/', ' ', function)
    
    func_declaration_match = re.search(patterns['function_declaration'], function)
    if func_declaration_match:
        start_index = func_declaration_match.end()
        function = function[start_index:]

    combined_pattern = '|'.join([pattern for key, pattern in patterns.items() if key != 'function_declaration'])
    compiled_pattern = re.compile(combined_pattern)

    matches = compiled_pattern.finditer(function)
    for match in matches:
        if match.group().strip():
            tokens.append(match.group().strip())

    return tokens

def process_c_file(file_path):
    """Process a single C file to extract and tokenize functions."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Define patterns for all, bad, and good functions
    all_functions_pattern = r'(void\s+\w+\(.*?{.*?})'
    bad_pattern = r'(void\s+(?:\w+_)?bad\(\).*?{.*?})'
    good_pattern = r'(void\s+(?:\w+_)?good\(\).*?{.*?})'


    # Extract functions
    all_functions = extract_functions(content, all_functions_pattern)
    bad_functions = extract_functions(content, bad_pattern)
    good_functions = extract_functions(content, good_pattern)

    # Process good functions: find and tokenize the functions they call
    tokenized_good = []
    for good_func in good_functions:
        called_funcs = extract_called_functions(good_func, all_functions)
        for func in called_funcs[1:]:
            tokenized_good.append(tokenize_function(func))

    # Tokenize bad functions
    tokenized_bad = [tokenize_function(func) for func in bad_functions]

    return tokenized_bad, tokenized_good

def vectorize_tokens(token_lists):
    all_tokens = [' '.join(tokens) for tokens in token_lists]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_tokens)
    return X.toarray()

def create_labeled_dataset(bad_vectors, good_vectors, test_size=0.2):
    max_len = 0
    if bad_vectors.size > 0:
        max_len = max(max_len, len(bad_vectors[0]))
    if good_vectors.size > 0:
        max_len = max(max_len, len(good_vectors[0]))

    if bad_vectors.size > 0:
        bad_vectors = np.array([np.pad(vec, (0, max_len - len(vec)), 'constant') for vec in bad_vectors])
    if good_vectors.size > 0:
        good_vectors = np.array([np.pad(vec, (0, max_len - len(vec)), 'constant') for vec in good_vectors])

    vectors = np.concatenate((bad_vectors, good_vectors)) if bad_vectors.size > 0 and good_vectors.size > 0 else np.array([])
    labels = [0] * len(bad_vectors) + [1] * len(good_vectors)

    if vectors.size > 0:
        X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = np.array([]), np.array([]), [], []

    return X_train, X_test, y_train, y_test

def pad_vectors(vectors, pad_length):
    """Pad the vectors to a specific length."""
    return np.array([np.pad(vec, (0, max(0, pad_length - len(vec))), 'constant') for vec in vectors])

def main(directory):
    """Process all C files in a given directory and aggregate data."""
    aggregated_bad_vectors = []
    aggregated_good_vectors = []

    for filename in os.listdir(directory):
        if filename.endswith(".c") or filename.endswith(".cpp"):
            file_path = os.path.join(directory, filename)
            bad, good = process_c_file(file_path)
            print(f"Processed {filename}:")

            if bad:
                aggregated_bad_vectors.extend(vectorize_tokens(bad))
            if good:
                aggregated_good_vectors.extend(vectorize_tokens(good))

    max_vector_length = max(len(max(aggregated_bad_vectors, key=len, default=[])),
                            len(max(aggregated_good_vectors, key=len, default=[])))

    aggregated_bad_vectors = pad_vectors(aggregated_bad_vectors, max_vector_length)
    aggregated_good_vectors = pad_vectors(aggregated_good_vectors, max_vector_length)

    X_train, X_test, y_train, y_test = create_labeled_dataset(aggregated_bad_vectors, aggregated_good_vectors)

    print("Training Data:", X_train)
    print("Training Labels:", y_train)
    print("Testing Data:", X_test)
    print("Testing Labels:", y_test)

if __name__ == "__main__":
    directory = './data/MemoryLeak/'
    main(directory)
