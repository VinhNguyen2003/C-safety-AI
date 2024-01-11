# C-safety-AI
## An AI model in training to detect common C pitfalls

This is my first data science project. I decided to take it on after completing CSCI 5521 - Machine Learning Fundamentals. I wanted to try to build and train an AI model to help with C programming as there are many pitfalls when it comes to using this language. Currently, the model is trained on data acquired from Juliet C/C++ which contains examples of pitfalls such as memory leaks, buffer overflow, etc.

The functions of a C program are tokenized and vectorized using existing models for Word2Vec, and then they are fed into a Tensorflow neural network model. After much time writing the scripts to preprocess the data and using them to train the model, I'm disappointed to say that the model sucks. Apparently everything is a memory leak to it except for some specific close to training data cases.

The next steps would be to implement feature selection to recognize certain patterns such as malloc() without free(), using unsafe functions such as fgets(), strcpy(), etc.

Even if the model sucks, and it does, I learned a lot when taking on this project. I will definitely come back to it from time to time to tweak and improve it.

## Sources and Attributions
All training and testing data are acquired from Juliet C/C++ 1.3 Test cases by the NSA Center for Assured Software 

Source: https://samate.nist.gov/SARD/test-suites/112