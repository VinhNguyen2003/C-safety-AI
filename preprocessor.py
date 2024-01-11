import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np

pitfall_mapping = {
    'Memory_Leak': 0,
    'Stack_Based_Buffer_Overflow': 1,
    'Heap_Based_Buffer_Overflow' : 2
    # more pitfalls, extend in the future
}

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
    # add comments to the features
    comment_tokens = re.findall(r'//.*?|/\*.*?\*/', function, re.DOTALL)
    comment_tokens = [token.strip() for token in comment_tokens]
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
    # tokenized comments are added here
    tokens.extend(comment_tokens)
    # remove the function name
    tokens = tokens[1:]
    return tokens

def process_c_file(file_path, pitfall_mapping):
    """Process a single C file to extract functions and their labels."""
    with open(file_path, 'r') as file:
        content = file.read()

    functions_with_labels = []
    for pitfall, label in pitfall_mapping.items():
        pattern = rf'(\w*{pitfall}\w*_bad\(\).*?{{.*?}})'
        functions = extract_functions(content, pattern)
        for func in functions:
            functions_with_labels.append((func, label))

    return functions_with_labels

def vectorize_function(function_tokens, model):
    vector = np.mean([model.wv[token] for token in function_tokens if token in model.wv], axis=0)
    return vector

def create_labeled_dataset(all_vectors, test_size=0.2):
    max_len = max([len(vec) for vec in all_vectors], default=0)

    padded_vectors = [np.pad(vec, (0, max_len - len(vec)), 'constant') for vec in all_vectors]
    vectors = np.array(padded_vectors)

    labels = [label for _, label in all_vectors]

    if vectors.size > 0:
        X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=test_size, random_state=42)
    else:
        X_train, X_test, y_train, y_test = np.array([]), np.array([]), [], []

    return X_train, X_test, y_train, y_test

def pad_vectors(vectors, pad_length):
    """Pad the vectors to a specific length."""
    return np.array([np.pad(vec, (0, max(0, pad_length - len(vec))), 'constant') for vec in vectors])

def preprocess_data(root_directory):
    all_tokenized_functions = []
    aggregated_functions_with_labels = [] 
    for directory in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, directory)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(".c") or filename.endswith(".cpp"):
                    file_path = os.path.join(dir_path, filename)
                    functions_with_labels = process_c_file(file_path, pitfall_mapping)
                    print(f"Processed {filename} in {directory}:")
                    for function, label in functions_with_labels:
                        tokenized_function = tokenize_function(function)
                        all_tokenized_functions.append(tokenized_function)
                        aggregated_functions_with_labels.append((function, label))

    # Train the Word2Vec model
    model = Word2Vec(sentences=all_tokenized_functions, vector_size=100, window=5, min_count=1, workers=4)
    model.save("./models/word2vec_model.model")
    # Vectorize functions using the trained model
    aggregated_vectors = []
    for function, label in aggregated_functions_with_labels:
        tokenized_function = tokenize_function(function)
        vectorized_function = vectorize_function(tokenized_function, model)
        aggregated_vectors.append((vectorized_function, label))

    # Split the data into training and testing sets
    X = [vector for vector, _ in aggregated_vectors]
    y = [label for _, label in aggregated_vectors]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    root_directory = './data'
    preprocess_data(root_directory)
