import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import ImageGrab, Image
import tkinter as tk
import win32gui

def load_mnist_model(model_path):
    model = models.load_model(model_path)
    return model

def preprocess_image(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    return img

def predict_digit(model, img):
    img = preprocess_image(img)
    res = model.predict(img)[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.model = model
        self.x = self.y = 0
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(self.model, im)
        self.label.configure(text=str(digit)+', '+str(int(acc*100))+'%')
        
    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')

if __name__ == "__main__":
    model_path = r'C:\Users\jaya2\Visual Code\ML\MiniProjects\HandwrittenDigitsRecognition\mnist_digit_recoginition.h5'
    mnist_model = load_mnist_model(model_path)
    app = App(mnist_model)
    app.mainloop()
