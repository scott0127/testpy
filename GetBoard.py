import cv2
import numpy as np


from ultralytics import YOLO
import cv2
import supervision as sv



# 讀取圖片
image = cv2.imread('./image/picture.jpg')

#load model 
model = YOLO('./models/yolov8m-world.pt')  

# Define custom classes
class_str=["person", "Chalkboard"]

model.set_classes(class_str)

bounding_box = sv.BoundingBoxAnnotator()

label_box = sv.LabelAnnotator()

mask_annotator = sv.MaskAnnotator()
results=model.predict(image,conf=0.5)
detection=sv.Detections.from_ultralytics(results[0])

        


#切出黑板
index=0
try:
    for id in detection.class_id:
        if id==1:
            x=int(detection.xyxy[index][0])
            y=int(detection.xyxy[index][1])
            h=int(detection.xyxy[index][3])-int(detection.xyxy[index][1])
            w=int(detection.xyxy[index][2])-int(detection.xyxy[index][0])
            
            annotate=image[y:y+h,x:x+w]
            print(detection.xyxy[index])
            break
        index+=1
except:
    pass

#cv2.imshow("yolopng", image)
cv2.waitKey(0)

#換成裁切成黑板的這樣就不會偵測矩形

# 將圖片轉換為灰度圖
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊降噪
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny邊緣檢測
edges = cv2.Canny(blurred, 50, 150)

# 圖像二值化處理
thresh = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 尋找輪廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化最大四邊形輪廓
max_quad = None
max_area = 0

# 遍歷所有輪廓
for contour in contours:
    # 將輪廓逼近為多邊形
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # 檢查是否近似為四邊形
    if len(approx) == 4:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_quad = approx
            max_area = area
            

# 如果存在四邊形輪廓
if max_quad is not None:
    # 將最大輪廓畫在黑板遮罩上
    x, y, w, h = cv2.boundingRect(max_quad)
    cv2.imwrite("thre.png", thresh[y:y+h,x:x+w])
    
    cv2.drawContours(image, [max_quad], -1, (255), thickness=5)
    
    
    #cv2.imshow("Board", image) #有需要的話可以顯示出來，就看的到openCV框出來的
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No quadrilateral found.")