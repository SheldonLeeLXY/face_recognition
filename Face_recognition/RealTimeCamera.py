import face_recognition
import cv2
import numpy as np
import time

# 打开摄像头
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()

while True:
    # 捕捉一帧视频
    ret, frame = video_capture.read()

    # 计算帧率
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # 将图像从 BGR 转换为 RGB
    rgb_frame = frame[:, :, ::-1]

    # 检测图像中的人脸位置和面部特征点
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    # 检查是否检测到人脸和面部特征
    for face_landmarks in face_landmarks_list:
        # 获取左右眼的坐标
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        nose_bridge = face_landmarks['nose_bridge']
        nose_tip = face_landmarks['nose_tip']
        chin = face_landmarks['chin']

        # 计算眼睛的中心点
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")
        nose_tip_center = np.mean(nose_tip, axis=0).astype("int")

        # 计算眼睛连线的角度（Roll 角）
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        roll_angle = np.degrees(np.arctan2(dy, dx))

        # 计算 Yaw 角度
        # 如果鼻子的位置比眼睛更靠右，则人脸朝右（Yaw > 0），否则朝左
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype("int")
        yaw_angle = np.degrees(np.arctan2(nose_tip_center[0] - eye_center[0], nose_tip_center[1] - eye_center[1]))

        # 计算 Pitch 角度
        # 调整计算，使 Pitch 角度更符合头部前倾和后仰
        chin_bottom = chin[8]  # 下巴最底部的点
        pitch_angle = np.degrees(np.arctan2(chin_bottom[1] - nose_tip_center[1], chin_bottom[0] - nose_tip_center[0]))
        pitch_angle = 90 - abs(pitch_angle)  # 将角度调整为符合直观的上下倾斜

        # 在图像上显示所有角度
        cv2.putText(frame, f"Roll Angle: {roll_angle:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw Angle: {yaw_angle:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch Angle: {pitch_angle:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示帧率
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 画出关键点
        cv2.circle(frame, tuple(left_eye_center), 2, (255, 0, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 2, (255, 0, 0), -1)
        cv2.circle(frame, tuple(nose_tip_center), 2, (0, 0, 255), -1)
        cv2.circle(frame, tuple(chin_bottom), 2, (0, 255, 255), -1)

    # 显示结果
    cv2.imshow('Face Rotation Detection with Yaw, Pitch, Roll', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
video_capture.release()
cv2.destroyAllWindows()
