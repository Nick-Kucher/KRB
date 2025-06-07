import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

model = YOLO(r"C:\Users\nickk\OneDrive\Desktop\Навчання\KRB\runs\detect\train\weights\best.pt")

os.makedirs("results", exist_ok=True)

root = tk.Tk()
root.title("🚗 Паркувальний детектор")
root.geometry("850x720")
root.configure(bg="#f7f7f7")

def load_gt_annotations(path, img_width, img_height):
    boxes, classes = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            x1 = int((xc - w / 2) * img_width)
            y1 = int((yc - h / 2) * img_height)
            x2 = int((xc + w / 2) * img_width)
            y2 = int((yc + h / 2) * img_height)
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return boxes, classes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

title = tk.Label(root, text="Система виявлення вільних та зайнятих паркомісць", font=("Arial", 18, "bold"), bg="#f7f7f7", fg="#333")
title.pack(pady=10)

param_frame = tk.Frame(root, bg="#f7f7f7")
param_frame.pack(pady=10)

tk.Label(param_frame, text="imgsz:", font=("Arial", 12), bg="#f7f7f7").grid(row=0, column=0, padx=5)
imgsz_entry = tk.Entry(param_frame, width=6)
imgsz_entry.insert(0, "960")
imgsz_entry.grid(row=0, column=1, padx=5)

tk.Label(param_frame, text="conf:", font=("Arial", 12), bg="#f7f7f7").grid(row=0, column=2, padx=5)
conf_entry = tk.Entry(param_frame, width=6)
conf_entry.insert(0, "0.1")
conf_entry.grid(row=0, column=3, padx=5)

tk.Label(param_frame, text="IoU:", font=("Arial", 12), bg="#f7f7f7").grid(row=0, column=4, padx=5)
iou_entry = tk.Entry(param_frame, width=6)
iou_entry.insert(0, "0.5")
iou_entry.grid(row=0, column=5, padx=5)

btn_frame = tk.Frame(root, bg="#f7f7f7")
btn_frame.pack(pady=15)

panel = tk.Label(root, bg="#e0e0e0", text="Тут буде відображено зображення", font=("Arial", 14, "italic"), fg="#999")
panel.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f7f7f7", fg="#444")
result_label.pack(pady=10)

metrics_label = tk.Label(root, text="", font=("Arial", 12), bg="#f7f7f7", fg="#333")
metrics_label.pack(pady=5)

progress_frame = tk.Frame(root, bg="#f7f7f7")
progress_frame.pack(pady=5)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100, length=500)
progress_bar.grid(row=0, column=0, padx=5)
progress_percent = tk.Label(progress_frame, text="0%", font=("Arial", 12), bg="#f7f7f7")
progress_percent.grid(row=0, column=1)

def update_progress(value):
    progress_var.set(value)
    progress_percent.config(text=f"{int(value)}%")
    root.update_idletasks()

def detect_parking_thread(file_path, imgsz, conf, iou_thresh):
    update_progress(10)
    results = model.predict(source=file_path, imgsz=imgsz, conf=conf, iou=iou_thresh)
    update_progress(40)

    image = cv2.imread(file_path)
    height, width = image.shape[:2]
    gt_path = file_path.replace(".jpg", ".txt").replace(".png", ".txt")

    if not os.path.exists(gt_path):
        metrics_label.config(text="⚠️ Не знайдено GT-файл: " + os.path.basename(gt_path))
        update_progress(100)
        return

    gt_boxes, gt_classes = load_gt_annotations(gt_path, width, height)
    pred_boxes, pred_classes = [], []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        pred_boxes.append([x1, y1, x2, y2])
        pred_classes.append(cls_id)
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    update_progress(60)

    y_true = []
    y_pred = []
    matched_pred = set()

    for gt_box, gt_cls in zip(gt_boxes, gt_classes):
        best_iou = 0
        best_idx = -1
        for idx, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
            if idx in matched_pred:
                continue
            iou = compute_iou(gt_box, pred_box)
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx != -1:
            y_true.append(gt_cls)
            y_pred.append(pred_classes[best_idx])
            matched_pred.add(best_idx)
        else:
            y_true.append(gt_cls)
            y_pred.append(2)  

    update_progress(80)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    result_label.config(text=f"✅ Вільних місць: {pred_classes.count(0)}    ⛔ Зайнятих місць: {pred_classes.count(1)}")
    metrics_label.config(
        text=f"📊 Метрики :\nAccuracy: {acc:.2f}   Precision: {prec:.2f}   Recall: {rec:.2f}   F1: {f1:.2f}")

    save_path = f"results/result_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(save_path, image)

    img = Image.open(save_path)
    img = img.resize((800, 400), Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk, text="")
    panel.image = img_tk
    update_progress(100)

def detect_parking():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    try:
        imgsz = int(imgsz_entry.get())
        conf = float(conf_entry.get())
        iou_thresh = float(iou_entry.get())
    except ValueError:
        result_label.config(text="❌ Некоректні параметри.")
        return
    update_progress(0)
    result_label.config(text="")
    panel.config(image="", text="Обробка...")
    metrics_label.config(text="")
    threading.Thread(target=detect_parking_thread, args=(file_path, imgsz, conf, iou_thresh), daemon=True).start()

def clear_all():
    panel.config(image="", text="Тут буде відображено зображення")
    panel.image = None
    result_label.config(text="")
    metrics_label.config(text="")
    update_progress(0)

btn_select = tk.Button(btn_frame, text="📂 Обрати зображення", command=detect_parking,
                       font=("Arial", 14), bg="#2196f3", fg="white", width=20)
btn_select.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(btn_frame, text="🪑 Очистити", command=clear_all,
                      font=("Arial", 14), bg="#ff5722", fg="white", width=12)
btn_clear.grid(row=0, column=1, padx=10)

root.mainloop()
