import face_recognition
import cv2

# 加载图片
image = face_recognition.load_image_file("obama-1080p.jpg")

# 检测图像中的人脸位置
face_locations = face_recognition.face_locations(image)

# 打印检测到的人脸数量
print(f"检测到 {len(face_locations)} 张人脸.")

# 检测五官位置（面部特征点）
face_landmarks_list = face_recognition.face_landmarks(image)

# 在图像中标记人脸位置
for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
    # 用矩形框标记人脸区域
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    # 标记五官位置
    for feature in face_landmarks:
        for point in face_landmarks[feature]:
            # 在五官位置上画点
            cv2.circle(image, point, 2, (255, 0, 0), -1)

# 显示结果
cv2.imshow("Face Detection with Landmarks", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
