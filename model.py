import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from preprocessor import preprocess_data, pitfall_mapping

X_train, X_test, y_train, y_test = preprocess_data('./data')

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(pitfall_mapping), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest Loss:', test_loss)
print('\nTest Accuracy:', test_acc)

model.save('./models/CSafetyModel.h5')

