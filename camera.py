from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO('older_models/x-ray-Segmentation.pt')

# 카메라 스트림
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 객체 감지
    results = model.predict(frame, stream=True)

    # 결과를 시각화
    for result in results:
        frame = result.plot()

    cv2.imshow("X-ray Yolo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
