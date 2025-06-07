import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

class PreprocessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🛠️ Попередня обробка зображень")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")

        self.image_path = None
        self.original_image = None
        self.processed_image = None

        self.panel = tk.Label(root, bg="#ddd")
        self.panel.pack(pady=10)

        control_frame = tk.Frame(root, bg="#f0f0f0")
        control_frame.pack()

        tk.Button(control_frame, text="📂 Відкрити зображення", command=self.load_image,
                  font=("Arial", 12), bg="#2196f3", fg="white").grid(row=0, column=0, padx=10)

        tk.Button(control_frame, text="💾 Зберегти", command=self.save_image,
                  font=("Arial", 12), bg="#4caf50", fg="white").grid(row=0, column=1, padx=10)

        slider_frame = tk.Frame(root, bg="#f0f0f0")
        slider_frame.pack(pady=10)

        tk.Label(slider_frame, text="Яскравість (alpha):", bg="#f0f0f0", font=("Arial", 10)).grid(row=0, column=0)
        self.alpha_slider = tk.Scale(slider_frame, from_=0.5, to=3.0, resolution=0.1,
                                     orient="horizontal", length=300, command=self.update_image)
        self.alpha_slider.set(1.2)
        self.alpha_slider.grid(row=0, column=1, padx=10)

        tk.Label(slider_frame, text="Контраст (beta):", bg="#f0f0f0", font=("Arial", 10)).grid(row=1, column=0)
        self.beta_slider = tk.Scale(slider_frame, from_=-100, to=100, resolution=1,
                                    orient="horizontal", length=300, command=self.update_image)
        self.beta_slider.set(20)
        self.beta_slider.grid(row=1, column=1, padx=10)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not self.image_path:
            return
        self.original_image = cv2.imread(self.image_path)
        self.update_image(None)

    def update_image(self, _):
        if self.original_image is None:
            return

        alpha = self.alpha_slider.get()
        beta = self.beta_slider.get()

        img = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        self.processed_image = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((800, 400), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.panel.config(image=img_tk)
        self.panel.image = img_tk

    def save_image(self):
        if self.processed_image is None:
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                 filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, self.processed_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = PreprocessingApp(root)
    root.mainloop()
