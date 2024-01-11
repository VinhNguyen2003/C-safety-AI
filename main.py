import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from preprocessor import tokenize_function, vectorize_function
import numpy as np

word2vec_model = Word2Vec.load("models/word2vec_model.model")
nn_model = load_model('models/CSafetyModel.h5')

def process_file(file_path, nn_model):
    with open(file_path, 'r') as file:
        content = file.read()

    tokenized_content = tokenize_function(content)
    vectorized_content = vectorize_function(tokenized_content, word2vec_model)

    vectorized_content = np.array([vectorized_content])

    prediction = nn_model.predict(vectorized_content)
    return np.argmax(prediction, axis=1)

def main():
    print("____________________________________________________")
    print("Welcome to the C Safety AI Model!")
    file_path = input("Enter the path of your .c or .cpp file: ")

    if not os.path.isfile(file_path) or not (file_path.endswith('.c') or file_path.endswith('.cpp')):
        print("Invalid file. Please enter a valid .c or .cpp file path.")
        return

    predicted_label = process_file(file_path, nn_model)
    predicted_label = int(predicted_label[0])

    pitfalls = {
        0: 'Memory Leak',
        1: 'Stack-Based Buffer Overflow',
        2: 'Heap-Based Buffer Overflow',
    }

    if predicted_label in pitfalls:
        print(f"The Model detected the following pitfall: {pitfalls[predicted_label]}")
    else:
        print("Based on the Model, your program is safe. However, do not trust a ML model trained by an amateur.")

if __name__ == "__main__":
    main()
