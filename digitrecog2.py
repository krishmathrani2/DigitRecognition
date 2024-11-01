#digitrecognition using the MNIST dataset
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
from PIL.Image import Resampling

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)

print("Training model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )
    ]
)

print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

def preprocess_image(image):
    image = image.convert('L')

    image = ImageOps.invert(image)

    image = image.resize((28, 28), Resampling.LANCZOS)

    image_arr = np.array(image, dtype=np.float32)
    image_arr = image_arr / 255.0

    return image_arr.reshape(1, 28, 28, 1)

def predict_digit(image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image, verbose=0)
        confidence = np.max(predictions) * 100
        digit = np.argmax(predictions)
        return digit, confidence
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognition")

        self.accuracy_label = tk.Label(
            self,
            text=f"Model Accuracy: {test_accuracy:.2%}",
            font=("Helvetica", 12)
        )
        self.accuracy_label.pack(pady=5)

        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(pady=5)

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side="left", padx=5)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.line_width = 20

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                  fill="black", width=self.line_width,
                                  capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y],
                          fill="black", width=self.line_width)
        self.last_x = x
        self.last_y = y

    def release(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        try:
            digit, confidence = predict_digit(self.image)
            self.show_prediction(digit, confidence)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_prediction(self, digit, confidence):
        result_window = tk.Toplevel(self)
        result_window.title("Prediction Result")

        processed_image = preprocess_image(self.image)
        processed_image_pil = Image.fromarray((processed_image.reshape(28, 28) * 255).astype(np.uint8))
        processed_image_pil = processed_image_pil.resize((140, 140), Resampling.NEAREST)
        processed_image_tk = ImageTk.PhotoImage(processed_image_pil)

        label_image = tk.Label(result_window, image=processed_image_tk)
        label_image.image = processed_image_tk
        label_image.pack(pady=10)

        tk.Label(
            result_window,
            text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%",
            font=("Helvetica", 16)
        ).pack(pady=10)

if __name__ == "__main__":
    try:
        from PIL import ImageOps  
        app = App()
        app.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
