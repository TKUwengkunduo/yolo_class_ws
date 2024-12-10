import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# Load a image
image = cv2.imread('intersection.jpg')

# Predict with the model
results = model(image)  # predict on an image


"""========================================================"""
img = results[0].plot()  # This plots the detections on the image
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 創建可調整大小的視窗
cv2.namedWindow('Detection Results', cv2.WINDOW_NORMAL)

# 顯示結果圖片
cv2.imshow('Detection Results', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""========================================================"""





"""================== U1 觀察results變數 ==================="""
"""========================================================"""
# print(results)
"""========================================================"""



"""================== U2 取得並繪製邊界框資料 ================"""
"""========================================================"""
# # 獲取檢測結果 (包括每個框的座標、置信度、以及類別標籤)
# detections = results[0].boxes.data.cpu().numpy()  # 獲取所有檢測框的數據

# # 逐一處理檢測結果
# for det in detections:
#     print(det)

#     x1, y1, x2, y2, conf, cls = det  # 解壓座標、置信度和類別
#     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 座標轉換為整數
#     label = f"{model.names[int(cls)]} {conf:.2f}"  # 類別名稱和置信度

#     # 繪製檢測框和標籤
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
#     cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
"""========================================================"""




"""==========================U3============================"""
"""========================================================"""
# # 創建可調整大小的視窗
# cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)

# # 顯示結果圖片
# cv2.imshow('YOLOv8 Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""========================================================"""




