from ultralytics import YOLO
import torch

def train_yolo_classifier():
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found, training will run on CPU.")

    model = YOLO("yolov12n.pt")

    model.train(
        data='dataset/data.yaml',  
        epochs=10,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        augment=True,
        name="parking_detector"
    )

if __name__ == "__main__":
    train_yolo_classifier()
