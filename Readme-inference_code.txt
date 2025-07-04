=======================
📦 MÔ TẢ FILE: model.py
=======================

🔹 Chức năng:
File này định nghĩa một class `Model` để chạy inference cho mô hình YOLOv8 dạng TFLite (TensorFlow Lite), phục vụ cho tác vụ phân đoạn đối tượng (instance segmentation).

---------------------
📚 Thư viện cần thiết:
---------------------
- numpy
- opencv-python
- tensorflow >= 2.x (đã bao gồm tensorflow.lite)

-------------------------------
⚙️ Hướng dẫn cài đặt môi trường:
-------------------------------
1. Tạo môi trường ảo (tuỳ chọn):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
````

2. Cài đặt thư viện:

   ```bash
   pip install numpy opencv-python tensorflow
   ```

---

## 🔍 Cách sử dụng (ví dụ đơn giản):

```python
from model import Model
from PIL import Image

model = Model("best_float32.tflite")   # Đường dẫn tới mô hình TFLite
image = Image.open("image.jpg")
results = model.predict(image)

for item in results:
    class_id, confidence, *polygon = item
    print(f"Class: {class_id}, Confidence: {confidence:.2f}, Polygon: {polygon[:6]}...")

## ⚠️ Lưu ý khi chạy mô hình:

* File `best_float32.tflite` là mô hình đã được huấn luyện và convert đúng chuẩn YOLOv8-seg dạng TFLite.
* Đầu vào ảnh cần là RGB, có thể là `PIL.Image` hoặc `numpy.ndarray`.
* Đầu ra là danh sách các polygon đại diện cho vật thể phân đoạn trong ảnh, kèm theo class và độ tin cậy.
