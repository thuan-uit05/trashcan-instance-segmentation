README.txt

## 📦 Danh sách các thư viện cần thiết

```
ultralytics
pycocotools
opencv-python
matplotlib
tqdm
PyYAML
```

Các thư viện hệ thống hoặc được sử dụng mặc định trên Google Colab:

```
tensorflow
numpy
shutil
glob
os
json
zipfile
```

---

## ⚙️ Hướng dẫn cài đặt và thiết lập môi trường

Bước 1: Cài đặt thư viện
```
pip install ultralytics pycocotools opencv-python matplotlib tqdm PyYAML
```

Bước 2: Upload dữ liệu và script
- Upload lần lượt:
  - File `original_data.zip` (chứa ảnh và annotation gốc)
  - File `trash_can_coco.py` (script chuyển đổi annotation COCO)
  - File `TrashCan.zip` (dữ liệu COCO dạng instance segmentation)

Bước 3: Tạo dataset chuẩn YOLOv8
Script sẽ tự động:
- Unzip dữ liệu
- Chuyển annotation COCO về định dạng YOLO instance segmentation
- Chuẩn hóa tên file ảnh `.jpg`
- Lọc bỏ ảnh không có annotation

Bước 4: Huấn luyện mô hình YOLOv8-Seg
- Mô hình sử dụng: `yolov8n-seg.pt`
- Epochs: 50
- Batch size: 64
- Image size: 640
```
yolo task=segment mode=train model=yolov8n-seg.pt data=/content/data.yaml epochs=50 imgsz=640 batch=64
```

Bước 5: Dự đoán trên tập validation
```
yolo task=segment mode=predict model=.../best.pt source=.../val/images
```

Bước 6: Xuất mô hình sang định dạng TFLite
```
yolo export model=.../best.pt format=tflite imgsz=320
```

---

## 📝 Các lưu ý khác

- Đảm bảo file JSON annotation có extension `.jpg` trùng khớp với ảnh thật.
- Không để ảnh không có annotation trong thư mục train/val.
- Mô hình xuất TFLite có thể dùng cho inference trên thiết bị nhúng (như Raspberry Pi hoặc Android).