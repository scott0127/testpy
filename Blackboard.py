import cv2
import numpy as np

from ultralytics import YOLO
import supervision as sv

class Blackboard:
    def __init__(self,image_path):
        self.board_image = cv2.imread(image_path)
        self.x=0
        self.y=0
        self.width=0
        self.height=0
        self.YOLO_image=None
    
    def run(self):
        try:
            self.Do_Object_Detection()
            self.GetBlackboard()
        except:
            pass
        
        pass
    
    def show(self):
        try:
            cv2.imshow("image",self.Final_image)
            cv2.waitKey(0)
        except:
            cv2.imshow("image",self.board_image)
            cv2.waitKey(0)
        pass
    
    def Do_Object_Detection(self):
        model = YOLO('./models/yolov8m-world.pt')
        class_str=["person", "Chalkboard"]
        model.set_classes(class_str)
        results=model.predict(self.board_image,conf=0.5)
        detection=sv.Detections.from_ultralytics(results[0])
        
        #切出偵測到的黑板，目前先當抓最大的，而且只有一個黑板
        index=0
        try:
            for id in detection.class_id:
                if id==1:
                    x=int(detection.xyxy[index][0])
                    y=int(detection.xyxy[index][1])
                    h=int(detection.xyxy[index][3])-int(detection.xyxy[index][1])
                    w=int(detection.xyxy[index][2])-int(detection.xyxy[index][0])
                    
                    annotate=self.board_image[y:y+h,x:x+w]
                    #print(detection.xyxy[index])
                    break
                index+=1
        except:
            return
        
        self.YOLO_image=annotate
        return
    
    def GetBlackboard(self):
        try:
            gray = cv2.cvtColor(self.YOLO_image, cv2.COLOR_BGR2GRAY)
        except:
            gray = cv2.cvtColor(self.board_image, cv2.COLOR_BGR2GRAY)
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
            cv2.imwrite("Output.png", thresh[y:y+h,x:x+w])
            cv2.drawContours(self.board_image, [max_quad], -1, (255), thickness=5)
            self.Final_image=thresh[y:y+h,x:x+w]
            #cv2.imshow("Board", image) #有需要的話可以顯示出來，就看的到openCV框出來的黑板
        else:
            print("No quadrilateral foundW.")
        pass