import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import uuid
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model = YOLO(r"C:\Users\nickk\OneDrive\Desktop\Навчання\KRB\runs\detect\train\weights\best.pt")

IMAGE_FOLDER = r"C:\Users\nickk\OneDrive\Desktop\Навчання\KRB\test"
RESULTS_FOLDER = "results"
ANNOTATIONS_FOLDER = r"C:\Users\nickk\OneDrive\Desktop\Навчання\KRB\test"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

root = tk.Tk()
root.title("🚗 Паркувальний детектор")
root.geometry("900x800")
root.configure(bg="#2e2e2e")

style = ttk.Style(root)
style.theme_use('clam')
style.configure("TLabel", background="#2e2e2e", foreground="#ddd", font=("Segoe UI", 11))
style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#f0f0f0")
style.configure("TButton", font=("Segoe UI", 12, "bold"), foreground="#222", background="#ccc")
style.map("TButton", background=[('active', '#448aff')], foreground=[('active', '#fff')])

top_frame = tk.Frame(root, bg="#2e2e2e")
top_frame.pack(fill="x", pady=(10, 5))

params_frame = tk.Frame(root, bg="#2e2e2e")
params_frame.pack(fill="x", pady=5)

params_inner_frame = tk.Frame(params_frame, bg="#2e2e2e")
params_inner_frame.pack(anchor="center")

tk.Label(params_inner_frame, text="Розмір зображення (imgsz):", bg="#2e2e2e", fg="#ddd").pack(side=tk.LEFT, padx=5)
imgsz_entry = tk.Entry(params_inner_frame, width=6)
imgsz_entry.insert(0, "1400")
imgsz_entry.pack(side=tk.LEFT, padx=5)

tk.Label(params_inner_frame, text="Поріг довіри (conf):", bg="#2e2e2e", fg="#ddd").pack(side=tk.LEFT, padx=5)
conf_entry = tk.Entry(params_inner_frame, width=6)
conf_entry.insert(0, "0.1")
conf_entry.pack(side=tk.LEFT, padx=5)

tk.Label(params_inner_frame, text="IoU поріг:", bg="#2e2e2e", fg="#ddd").pack(side=tk.LEFT, padx=5)
iou_entry = tk.Entry(params_inner_frame, width=6)
iou_entry.insert(0, "0.4")
iou_entry.pack(side=tk.LEFT, padx=5)

button_frame = tk.Frame(root, bg="#2e2e2e")
button_frame.pack(fill="x", pady=(0, 5))

button_inner_frame = tk.Frame(button_frame, bg="#2e2e2e")
button_inner_frame.pack(anchor="center")

def open_file_dialog():
    path = filedialog.askopenfilename(initialdir=IMAGE_FOLDER, filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        start_detection(path)

btn_gallery = ttk.Button(button_inner_frame, text="📁 Відкрити галерею", command=lambda: show_gallery())
btn_gallery.pack(side=tk.LEFT, padx=20)

btn_open = ttk.Button(button_inner_frame, text="🖼 Обрати інше зображення", command=open_file_dialog)
btn_open.pack(side=tk.LEFT, padx=20)

gallery_frame = tk.Frame(root, bg="#2e2e2e")
gallery_frame.pack(fill='both', expand=True)

detect_frame = tk.Frame(root, bg="#2e2e2e")
detect_frame.pack_forget()

title_label = ttk.Label(top_frame, text="🚗 Виявлення вільних паркомісць", style="Header.TLabel")
title_label.pack()

gallery_images = []
current_image_path = None

def clear_gallery():
    for widget in gallery_frame.winfo_children():
        widget.destroy()
    gallery_images.clear()

def show_gallery():
    clear_gallery()
    detect_frame.pack_forget()
    gallery_frame.pack(fill='both', expand=True)

    files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    row_frame = None
    for idx, file in enumerate(files):
        if idx % 5 == 0:
            row_frame = tk.Frame(gallery_frame, bg="#2e2e2e")
            row_frame.pack(fill='x', pady=15)

        path = os.path.join(IMAGE_FOLDER, file)
        img = Image.open(path)
        img.thumbnail((150, 150))
        img_tk = ImageTk.PhotoImage(img)

        btn = tk.Button(row_frame, image=img_tk, borderwidth=0,
                        command=lambda p=path: start_detection(p))
        btn.image = img_tk
        btn.pack(side=tk.LEFT, padx=15, pady=10)
        gallery_images.append(btn)

def load_and_detect(img_path):
    for w in detect_frame.winfo_children():
        w.destroy()

    back_btn = ttk.Button(detect_frame, text="⬅ Назад", command=lambda: (detect_frame.pack_forget(), show_gallery()))
    back_btn.pack(anchor="nw", padx=10, pady=10)

    panel = tk.Label(detect_frame, bg="#444", text="Завантаження зображення...", fg="#eee", font=("Segoe UI", 14, "italic"))
    panel.pack(pady=10)

    result_label = tk.Label(detect_frame, text="", bg="#2e2e2e", fg="#ddd", font=("Segoe UI", 14))
    result_label.pack(pady=6)

    metrics_label = tk.Label(detect_frame, text="", bg="#2e2e2e", fg="#ccc", font=("Segoe UI", 12))
    metrics_label.pack(pady=4)

    def detect_parking_thread():
        try:
            imgsz = int(imgsz_entry.get())
            conf = float(conf_entry.get())
            iou_thr = float(iou_entry.get())
        except Exception:
            result_label.config(text="❌ Некоректні параметри")
            return

        results = model.predict(source=img_path, imgsz=imgsz, conf=conf, iou=iou_thr)
        frame = cv2.imread(img_path)
        img_h, img_w = frame.shape[:2]

        pred_boxes = []
        free, occupied = 0, 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            pred_boxes.append((cls_id, [x1, y1, x2, y2]))
            color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
            if cls_id == 0:
                free += 1
            else:
                occupied += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        save_path = os.path.join(RESULTS_FOLDER, f"result_{uuid.uuid4().hex[:8]}.jpg")
        cv2.imwrite(save_path, frame)

        img = Image.open(save_path)
        img = img.resize((800, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)

        panel.config(image=img_tk, text="")
        panel.image = img_tk

        result_label.config(text=f"✅ Вільних місць: {free}    ⛔ Зайнятих місць: {occupied}")

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        gt_txt_path = os.path.join(ANNOTATIONS_FOLDER, base_name + ".txt")

        if os.path.isfile(gt_txt_path):
            gt_boxes = read_gt_labels(gt_txt_path, img_w, img_h)

            if len(gt_boxes) > 0:
                y_true, y_pred = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5)

                if len(y_true) > 0:
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
                    rec = recall_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='binary', pos_label=0, zero_division=0)
                    metrics_label.config(
                        text=f"📊 Метрики:  Accuracy={acc:.2f}  Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}")

    threading.Thread(target=detect_parking_thread, daemon=True).start()

def start_detection(image_path):
    gallery_frame.pack_forget()
    detect_frame.pack(fill='both', expand=True)
    load_and_detect(image_path)

def read_gt_labels(path, img_w, img_h):
    gt = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            cls = int(cls)
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            gt.append((cls, [x1, y1, x2, y2]))
    return gt

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5):
    y_true = []
    y_pred = []
    matched_pred = set()

    for gt_cls, gt_box in gt_boxes:
        best_iou = 0
        best_pred = None
        for idx, (pred_cls, pred_box) in enumerate(pred_boxes):
            if idx in matched_pred:
                continue
            iou_val = iou(gt_box, pred_box)
            if iou_val > best_iou:
                best_iou = iou_val
                best_pred = (pred_cls, idx)
        if best_iou >= iou_threshold:
            pred_cls, pred_idx = best_pred
            y_true.append(gt_cls)
            y_pred.append(pred_cls)
            matched_pred.add(pred_idx)
        else:
            y_true.append(gt_cls)
            y_pred.append(1 - gt_cls)  
    return y_true, y_pred

show_gallery()
root.mainloop()
