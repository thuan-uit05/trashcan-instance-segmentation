# trashcan-instance-segmentation

YOLOv8 Instance Segmentation for marine trash detection using TrashCAN 1.0 dataset, exported to TensorFlow Lite for edge deployment.  
Đồ án thực hiện bởi **Thái Trường Thi & Trương Gia Thuận**.

---

## 📜 Mô tả dự án

Đây là dự án đồ án xây dựng mô hình phân đoạn instance (instance segmentation) để phát hiện **rác thải dưới môi trường biển**, sử dụng tập dữ liệu **TrashCAN 1.0** (COCO Polygon format).

Mô hình được huấn luyện bằng YOLOv8-Seg trên Google Colab GPU T4 và được chuyển đổi sang TensorFlow Lite để dễ dàng triển khai trên thiết bị nhúng (Raspberry Pi, Android).

---

## 🚀 Tính năng chính

✅ Phát hiện & phân đoạn (mask) các loại rác thải biển với 22 lớp  
✅ Sử dụng YOLOv8-Seg hiện đại, tốc độ nhanh (~14 FPS trên T4)  
✅ Chuyển đổi sang TensorFlow Lite (`.tflite`) cho inference trên thiết bị edge  
✅ Tích hợp sẵn script Python chạy TFLite để dự đoán & xuất polygon

---

## 📂 Cấu trúc dự án

```
.
├── TrashCan.ipynb           # Notebook huấn luyện & xuất model
├── best_float32.tflite      # Mô hình YOLOv8-Seg đã xuất sang TFLite (float32)
├── class_names.txt          # Mapping tên lớp
├── model.py                 # Script chạy inference TFLite (xuất polygon)
├── Readme.txt               # Hướng dẫn cài đặt chi tiết & pipeline
├── 22521379_22521449.docx    # Báo cáo đồ án hoàn chỉnh
```

---

## ⚙️ Hướng dẫn cài đặt & chạy thử

### 1️⃣ Cài đặt thư viện
```bash
pip install ultralytics pycocotools opencv-python matplotlib tqdm PyYAML tensorflow
```

*(có thể dùng virtualenv nếu muốn)*

---

### 2️⃣ Chạy inference với model TFLite
```python
from model import Model
from PIL import Image

model = Model("best_float32.tflite")
img = Image.open("test_image.jpg")
results = model.predict(img)

for item in results:
    class_id, confidence, *polygon = item
    print(f"Class: {class_id}, Confidence: {confidence:.2f}, Polygon points: {polygon[:6]}...")
```

---

### 3️⃣ Thông tin model
- 🧠 Mô hình YOLOv8-Seg
- Huấn luyện 50 epochs, batch size 64, image size 640
- Chuyển export TFLite: 320x320 (float32)

---

## 📝 Dataset & class mapping

Dữ liệu: [TrashCAN 1.0](https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7)  
Annotation: COCO Polygon ➔ YOLOv8-Seg ➔ TFLite

Danh sách lớp (ví dụ):
```
0 : rov
1 : plant
2 : animal_fish
...
20 : trash_rope
21 : trash_net
```
(đầy đủ xem trong `class_names.txt`)

---

## 📊 Kết quả thực nghiệm

| Metric   | Giá trị |
|----------|---------|
| mAP50    | 0.756   |
| mAP50-95 | 0.483   |
| F1 Score | 0.323   |
| Precision| 0.505   |
| Recall   | 0.259   |
| FPS      | ~14     |

Chi tiết xem trong file `22521379_22521449.docx` (chứa đồ thị mAP, F1 và phân tích).

---

## 📌 Lưu ý triển khai

- File `model.py` đã tối ưu cho TensorFlow Lite, xuất polygon mask.
- Input cần ảnh RGB (`PIL.Image` hoặc `numpy.ndarray`).
- Output là list `[class_id, confidence, x1, y1, ..., xn, yn]`.

---

## 📜 License

MIT License.  
(Thêm LICENSE nếu open-source.)

---

## ✍️ Nhóm thực hiện
- **Thái Trường Thi**: Huấn luyện mô hình YOLOv8-Seg, chuẩn hóa dữ liệu, đánh giá kết quả.
- **Trương Gia Thuận**: Xử lý dữ liệu COCO ➔ YOLO, export TFLite, viết script inference, tổng hợp báo cáo.

---

🚀 **Enjoy!**  
> Nếu bạn thấy repo này hữu ích, hãy ⭐ Star để ủng hộ nhóm nhé!
