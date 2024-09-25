import face_recognition
import cv2
import numpy as np
import time

# 打开摄像头
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = time.time()
max_eye_distance = 0  # 初始化最大眼睛距离为0

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
        top_lip = face_landmarks['top_lip']
        bottom_lip = face_landmarks['bottom_lip']

        # 计算眼睛的中心点
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")
        eye_center = np.mean([left_eye_center, right_eye_center], axis=0).astype("int")

        # 计算鼻尖的中心点
        nose_tip_center = np.mean(nose_tip, axis=0).astype("int")

        # 计算 Yaw 角度
        dx = nose_tip_center[0] - eye_center[0]
        dy = nose_tip_center[1] - eye_center[1]
        yaw_angle = np.degrees(np.arctan2(dx, dy))

        # 计算两眼之间的水平距离
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)

        # 更新最大眼睛距离
        if eye_distance > max_eye_distance:
            max_eye_distance = eye_distance

        # 计算当前眼睛距离和最大眼睛距离的夹角
        if max_eye_distance > 0:
            cos_theta = eye_distance / max_eye_distance
            # 为了防止浮点精度问题导致cos_theta超出[-1, 1]范围
            cos_theta = np.clip(cos_theta, -1, 1)
            angle_between_distances = np.degrees(np.arccos(cos_theta))
        else:
            angle_between_distances = 0

        # 在图像上显示当前的眼睛距离和最大眼睛距离
        cv2.putText(frame, f"Eye Distance: {eye_distance:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Max Eye Distance: {max_eye_distance:.2f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Angle: {angle_between_distances:.2f} degrees", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 在图像上显示 Yaw 角度
        cv2.putText(frame, f"Yaw Angle: {yaw_angle:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示帧率
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 画出每个面部关键点
        for landmark, points in face_landmarks.items():
            for point in points:
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        # # 画出眼睛的中心点
        # cv2.circle(frame, tuple(left_eye_center), 4, (255, 0, 0), -1)
        # cv2.circle(frame, tuple(right_eye_center), 4, (255, 0, 0), -1)
        # cv2.circle(frame, tuple(nose_tip_center), 4, (255, 0, 0), -1)

    # 显示结果
    cv2.imshow('Face Rotation Detection with Yaw, Pitch, Roll', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
video_capture.release()
cv2.destroyAllWindows()
