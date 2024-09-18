import face_recognition
import cv2

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 捕捉一帧视频
    ret, frame = video_capture.read()

    # 将图像从 BGR 转换为 RGB，face_recognition 需要 RGB 图像
    rgb_frame = frame[:, :, ::-1]

    # 检测图像中的人脸位置
    face_locations = face_recognition.face_locations(rgb_frame)

    # 检测五官位置（面部特征点）
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    # 在图像中标记人脸位置
    for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
        # 用矩形框标记人脸区域
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # 标记五官位置
        for feature in face_landmarks:
            for point in face_landmarks[feature]:
                # 在五官位置上画点
                cv2.circle(frame, point, 2, (255, 0, 0), -1)

    # 显示结果
    cv2.imshow('Real-Time Face Detection with Landmarks', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
video_capture.release()
cv2.destroyAllWindows()
