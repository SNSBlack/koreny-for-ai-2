
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes

# Параметры
DATA_DIR = "edu_work"  # Папка с данными для обучения
OUTPUT_VIDEO_PATH = "output_video.mp4"
INPUT_VIDEO_PATH = "path/to/input_video.mp4"
NUM_CLASSES = 2  # Количество классов (включая фон)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели
def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Замена классификатора
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model.to(DEVICE)

# Подготовка данных
def prepare_data(data_dir):
    # Здесь можно реализовать логику для загрузки и подготовки данных
    # Короче нужен torchvision.datasets для создания кастомного датасета
    pass

# Обучение модели
def train_model(model, data_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for images, targets in data_loader:
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            # Обратное распространение ошибки
            losses.backward()
            # Здесь можно добавить оптимизатор и шаг обновления
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item()}")

# Обработка видео
def process_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    model.eval()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break


            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
            predictions = model(img_tensor)

            # Обработка предсказаний
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            for box, score in zip(boxes, scores):
                if score > 0.5:  # Порог уверенности
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    data_loader = prepare_data(DATA_DIR)
    model = load_model(NUM_CLASSES)
    train_model(model, data_loader)
    process_video(model, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)