import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
from PIL.Image import Resampling
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

digits = load_digits()
X, y = digits.data, digits.target

X = X.reshape((len(X), 8, 8, 1)) / 16.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
)
datagen.fit(X_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest',
)
datagen.fit(X_train)

model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_test, y_test))



def preprocess_image(image, threshold=127):  
    image = image.resize((8, 8), Resampling.LANCZOS)
    image = image.convert("L")
    image_arr = np.array(image)
    image_arr = np.where(image_arr > threshold, 0, 1) 
    return image_arr.reshape(1, 8, 8, 1)

def predict_digit(image):
    image = image.convert("L") 
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")

        self.canvas = tk.Canvas(self, width=200, height=200, bg="white")
        self.canvas.pack()
        
        self.button_frame = tk.Frame(self)
        self.button_frame.pack()

        self.predict_button = tk.Button(self.button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side="left")

        self.clear_button = tk.Button(self.button_frame, text="Clear", command=self.clear)
        self.clear_button.pack(side="left")

        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("RGB", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)  
        x2, y2 = (event.x + 5), (event.y + 5)
        self.line_width = 10 
        self.canvas.create_line(x1, y1, x2, y2, fill="black", width=self.line_width)
        self.draw.line([x1, y1, x2, y2], fill="black", width=self.line_width)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 200, 200), fill="white")

    def predict(self):
        prediction = predict_digit(self.image)
        self.show_prediction(prediction)

    def show_prediction(self, prediction):
        result_window = tk.Toplevel(self)
        result_window.title("Prediction")

        processed_image = preprocess_image(self.image)
        processed_image_pil = Image.fromarray((processed_image.reshape(8, 8) * 255).astype(np.uint8))
        processed_image_tk = ImageTk.PhotoImage(processed_image_pil)
        label_image = tk.Label(result_window, image=processed_image_tk)
        label_image.image = processed_image_tk  
        label_image.pack()

        tk.Label(result_window, text=f"Predicted Digit: {prediction}", font=("Helvetica", 24)).pack()

if __name__ == "__main__":
    app = App()
    app.mainloop()
