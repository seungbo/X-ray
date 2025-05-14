import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
torch.classes.__path__ = []

import io
import json
from typing import Any

import cv2
import pandas as pd
import time

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

import streamlit as st
st.set_page_config(page_title="X-RayVision", layout="wide")

# 공유 데이터 파일 경로 정의
SHARED_DATA_FILE = "shared_data.json"

# 기본 데이터 구조
DEFAULT_SHARED_DATA = {'current_counts': {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}, 'cumulative_counts': {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}, 'log_messages': []}

class Inference:
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.enable_trk = False  # Flag to toggle object tracking
        self.conf = 0.40  # Confidence threshold for detection
        self.iou = 0.30  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.warning_placeholder = None # Placeholder for hazard warnings
        self.counts_placeholder = None
        self.vid_file_name = None  # Video file name or webcam index
        self.selected_ind = []  # List of selected class indices for detection
        self.model = None  # YOLO model instance

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]
            LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def initialize_shared_data(self):
        """공유 JSON 파일을 기본값으로 초기화합니다."""
        try:
            need_init = True
            if os.path.exists(SHARED_DATA_FILE):
                with open(SHARED_DATA_FILE, 'r') as f:
                    try:
                        data = json.load(f)
                        if data == DEFAULT_SHARED_DATA:
                            need_init = False
                    except Exception:
                        pass
            if need_init:
                with open(SHARED_DATA_FILE, 'w') as f:
                    json.dump(DEFAULT_SHARED_DATA, f, indent=4)
                LOGGER.info(f"Initialized {SHARED_DATA_FILE}")
                print("Initialized shared_data.json")  # 실제로 초기화될 때만 출력
        except Exception as e:
            LOGGER.error(f"Failed to initialize {SHARED_DATA_FILE}: {e}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF8000; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">AI 기반 X-ray 위험 물품 자동 탐지 솔루션</h1></div>"""

        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.counts_placeholder = self.st.empty()

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            col1, col2, col3 = self.st.columns([1, 2, 1])
            with col2:
                logo = "scan.png"
                self.st.image(logo, width=150)

        self.st.sidebar.title("사용자 구성")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "영상 소스",
            ("webcam", "video"),
            index=None,                         # 초기 선택 없음
            placeholder="Choose an option",    # 안내 문구 표시 :contentReference[oaicite:0]{index=0}
        )
        self.enable_trk = self.st.sidebar.radio("객체 추적 여부", ("Yes", "No"))  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("신뢰도 임계값", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU 임계값", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        self.vid_file_name = ""
        self.webcam_stream = None
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open("ultralytics.mp4", "wb") as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            # webcam 모드에서는 아무런 안내 문구나 카메라 입력창을 표시하지 않음
            pass

    def configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection, including local 'models' folder
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        local_models = []
        if os.path.isdir(model_dir):
            # collect raw filenames with extensions
            for fname in os.listdir(model_dir):
                if fname.endswith((".pt", ".onnx")) and not fname.endswith("-obb.pt"):
                    local_models.append(os.path.splitext(fname)[0])
        # Combine local and GitHub asset models, sort alphabetically, and remove duplicates
        available_models = sorted(set(
            name for name in (
                local_models +
                [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
            )
            if not name.lower().endswith("-obb")
        ), reverse=True)

        # 영상 소스가 선택되지 않았으면 모델 선택 옵션 비활성화
        if not self.source:
            available_models = []

        # video 모드일 때 models 폴더에 있는 항목만 남기고 나머지는 숨김 처리
        if self.source == "video":
            available_models = sorted(set(local_models), reverse=True)

        # webcam 모드일 때 models 폴더와 YOLO11n으로 시작하는 모델만 보이게 필터링
        if self.source == "webcam":
            available_models = [
                m for m in available_models
                if m in local_models or m.lower().startswith("yolo11n")
            ]

        # Prioritize a model passed via command-line
        if self.model_path:
            custom_model = os.path.splitext(self.model_path)[0]
            if custom_model in available_models:
                available_models.remove(custom_model)
            available_models.insert(0, custom_model)
        # selected_model = self.st.sidebar.selectbox("Model", available_models)
        selected_model = self.st.sidebar.selectbox(
            "모델",
            available_models,
            index=None,                         # 초기 선택 없음
            placeholder="Choose an option",    # 안내 문구 표시
        )
        if selected_model:
            with self.st.spinner("Model is downloading..."):
                # Resolve the actual model file path, preferring local 'models' folder
                model_dir = os.path.join(os.path.dirname(__file__), "models")
                # Determine filename with extension
                model_name = selected_model
                if model_name and not model_name.lower().endswith((".pt", ".onnx")):
                    model_name = f"{model_name.lower()}.pt"
                # Local path in 'models' folder
                local_path = os.path.join(model_dir, model_name)
                # Use local file if exists, otherwise fallback to standard lookup
                load_path = local_path if os.path.exists(local_path) else model_name
                self.model = YOLO(load_path)
                class_names = sorted(self.model.names.values())
        else:
            class_names = []
        
        self.success_placeholder = self.st.empty()
        if selected_model:
            self.success_placeholder.success("Model loaded successfully!")

        if selected_model:
            all_classes_option = "All Classes"
            # -pose로 끝나는 모델이면 All Classes 옵션 제거
            if selected_model.lower().endswith("-pose"):
                options = class_names
            else:
                options = [all_classes_option] + class_names
        else:
            options = []
        selected_classes = self.st.sidebar.multiselect(
            "객체 종류",
            options,
            default=[],
            placeholder="Choose an option",
        )
        if selected_model:
            if all_classes_option in selected_classes:
                self.selected_ind = list(range(len(class_names)))
            else:
                self.selected_ind = []
                for option in selected_classes:
                    for idx, name in self.model.names.items():
                        if name == option:
                            self.selected_ind.append(idx)
                            break
        else:
            self.selected_ind = []

        if not isinstance(self.selected_ind, list): 
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""

        def flip_frame(frame):
            """프레임을 좌우 반전시키는 함수"""
            return cv2.flip(frame, 1)

        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app

        category_map = {
            "일반물품": [
                "Adapter", "Auto-lead-leash", "Baseball-glove", "Battery", "Belt", "Bolt", "Boots"
            ],
            "위해물품": [
                "Arrow-tip", "Awl", "Ax", "Baton-folding", "Big-ball", "Billiard-ball", "Bolt-cutter"
            ],
            "정보저장매체": [
                "CD", "Camcorder", "Camera", "Film", "Floppy-disk", "Folder-phone", "Hard-disk", "LP"
            ]
        }

        if self.st.sidebar.button("Start"):
            # Start 버튼 클릭 시 "Model loaded successfully!" 메시지 숨기기
            self.success_placeholder.empty()
            
            # Start 버튼 클릭 시 공유 데이터 파일 초기화
            self.initialize_shared_data()
 
            # 위해물품 경고 메시지를 표시할 위치 확보
            self.warning_placeholder = self.st.empty()
 
            row1 = self.st.columns(2)  # 상단 행 - 영상용
            
            row1[0].markdown("<h3 style='text-align: center;'>원본</h3>", unsafe_allow_html=True)
            row1[1].markdown("<h3 style='text-align: center;'>결과</h3>", unsafe_allow_html=True)
            
            self.org_frame = row1[0].empty()
            self.ann_frame = row1[1].empty()
            
            # 버튼 영역: Stop 버튼을 왼쪽에 배치
            btns = self.st.columns([1,1])
            stop_button = btns[0].button("Stop")
            log_messages_buffer = []  # 로컬 버퍼 사용
            
            self.counts_placeholder.empty()
            
            # 비디오 소스 설정
            if self.source == "webcam":
                cap = cv2.VideoCapture(0)  # 웹캠
            else:
                cap = cv2.VideoCapture(self.vid_file_name)  # 비디오 파일
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 최소화 (실시간성 향상)

            if not cap.isOpened():
                self.st.error("Could not open video source.")
                return

            # 누적 카운트 초기화
            cumulative_counts = {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}
            
            # 트래킹 사용 여부는 사용자 설정에 따름
            use_tracking = self.enable_trk == "Yes"
            
            # 알림 보낸 트래킹 ID 저장용 세트 (트래킹 사용 시)
            warned_track_ids = set()
            
            # 트래킹 ID 세트 초기화 (트래킹 사용 여부와 상관없이 항상 초기화)
            tracked_ids = set()
            
            # 트래킹 ID별 최초 감지 시각 저장 (트래킹 사용 시)
            track_id_first_seen = dict()
            
            # 감지된 객체 정보 저장 딕셔너리
            detected_objects = {}
            
            # 공유 데이터 저장을 위한 변수
            shared_data = DEFAULT_SHARED_DATA.copy()

            while True:
                success, frame = cap.read()
                if not success:
                    if self.source == "webcam":
                        continue  # 웹캠의 경우 다음 프레임 시도
                    else:
                        self.st.warning("Failed to read frame from video source.")
                        break

                # 웹캠 모드일 때만 프레임 반전
                if self.source == "webcam":
                    frame = flip_frame(frame)

                # Process frame with model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                with torch.no_grad():
                    if use_tracking:
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True, device=device
                        )
                    else:
                        results = self.model(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, device=device
                        )
                annotated_frame = results[0].plot()

                # 현재 프레임에서 위해물품 감지 여부 플래그
                hazard_detected_in_frame = False

                # 현재 프레임에서 감지된 객체 카운트
                current_counts = {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}
                
                # 트래킹 사용하는 경우
                if use_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    now = time.time()  # 현재 시간
                    for i, det in enumerate(results[0].boxes):
                        track_id = int(det.id.item()) if det.id is not None else None
                        cls_id = int(det.cls.item())
                        cls_name = self.model.names[cls_id]
                        
                        # 로그 메시지 생성 및 저장
                        log_msg = f"탐지: {cls_name} (신뢰도: {det.conf.item():.2f})"
                        LOGGER.info(log_msg)

                        log_messages_buffer.append(log_msg)
                        log_messages_buffer = log_messages_buffer[-10:] # 최신 10개 유지

                        # 위해물품 카테고리 확인 및 중복 알림 방지
                        if any(cls_name in items for cat, items in category_map.items() if cat == "위해물품"):
                            if track_id is not None and track_id not in warned_track_ids:
                                warned_track_ids.add(track_id) # 알림 보낸 ID 기록
                            hazard_detected_in_frame = True # 위해물품 감지 플래그 설정

                        # 현재 프레임 카운트 업데이트
                        for cat, items in category_map.items():
                            if cls_name in items:
                                current_counts[cat] += 1
                                break
                        
                        # 트래킹 ID가 있고 1초 이상 감지된 경우에만 누적 카운트 증가
                        if track_id is not None:
                            if track_id not in track_id_first_seen:
                                track_id_first_seen[track_id] = now
                            elif (track_id not in tracked_ids) and (now - track_id_first_seen[track_id] >= 1.0):
                                tracked_ids.add(track_id)
                                for cat, items in category_map.items():
                                    if cls_name in items:
                                        cumulative_counts[cat] += 1
                                        break
                else:
                    # 트래킹 사용하지 않는 경우
                    if results[0].boxes is not None:
                        for det in results[0].boxes:
                            cls_id = int(det.cls.item())
                            cls_name = self.model.names[cls_id]
                            box = det.xyxy.cpu().numpy()[0]  # xyxy 형식의 바운딩 박스

                            # 로그 메시지 생성 및 저장 (트래킹 미사용 시)
                            log_msg = f"탐지: {cls_name} (신뢰도: {det.conf.item():.2f})"
                            LOGGER.info(log_msg)
                        
                            log_messages_buffer.append(log_msg)
                            log_messages_buffer = log_messages_buffer[-10:] # 최신 10개 유지

                            # 위해물품 카테고리 확인 (트래킹 미사용 시, new_object 조건은 아래에서 확인)
                            is_hazard = any(cls_name in items for cat, items in category_map.items() if cat == "위해물품")

                            x1, y1, x2, y2 = box
                            
                            # 객체 정보 생성 (클래스 ID, 위치, 크기)
                            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            box_size = ((x2 - x1), (y2 - y1))
                            obj_info = (cls_id, box_center, box_size)
                            
                            # 현재 프레임 카운트 업데이트

                            for cat, items in category_map.items():
                                if cls_name in items:
                                    # cumulative_counts[cat] += 1
                                    current_counts[cat] += 1
                                    break
                            # 이 클래스의 객체 정보가 없으면 딕셔너리 초기화
                            if cls_name not in detected_objects:
                                detected_objects[cls_name] = []
                                # 이 클래스가 처음 감지된 경우 누적 카운트 증가
                                for cat, items in category_map.items():
                                    if cls_name in items:
                                        cumulative_counts[cat] += 1
                                        break
                            
                            # 기존에 감지된 유사한 객체가 있는지 확인
                            new_object = True
                            if cls_name in detected_objects: # Ensure key exists before iterating
                                for stored_obj in detected_objects[cls_name]:
                                    stored_cls_id, stored_center, stored_size = stored_obj
                                    
                                    distance = ((box_center[0] - stored_center[0])**2 + 
                                            (box_center[1] - stored_center[1])**2)**0.5
                                    
                                    size_ratio_w = box_size[0] / stored_size[0] if stored_size[0] > 0 else float('inf')
                                    size_ratio_h = box_size[1] / stored_size[1] if stored_size[1] > 0 else float('inf')
                                    
                                    if (distance < min(box_size[0], box_size[1]) * 0.5 and
                                        0.5 <= size_ratio_w <= 2.0 and
                                        0.5 <= size_ratio_h <= 2.0):
                                        new_object = False
                                        break
                            
                            if new_object:
                                if cls_name not in detected_objects: # Ensure list is initialized
                                     detected_objects[cls_name] = []
                                detected_objects[cls_name].append(obj_info)
                                if is_hazard:
                                    hazard_detected_in_frame = True

                current_df = pd.DataFrame({
                    '카테고리': list(current_counts.keys()), 
                    'count': list(current_counts.values())
                })
                cumulative_df = pd.DataFrame({
                    '카테고리': list(cumulative_counts.keys()),
                    'count': list(cumulative_counts.values())
                })

                # 프레임 처리 후 상태 표시기 업데이트
                if hazard_detected_in_frame:
                    self.warning_placeholder.error("🚨 상태: 위해물품 감지됨!", icon="🔥")
                else:
                    self.warning_placeholder.success("✅ 상태: 안전", icon="👍")

                # 공유 데이터 업데이트
                shared_data = {
                    'current_counts': current_counts,
                    'cumulative_counts': cumulative_counts,
                    'log_messages': log_messages_buffer
                }
                # JSON 파일에 쓰기
                try:
                    with open(SHARED_DATA_FILE, 'w') as f:
                        json.dump(shared_data, f, indent=4)
                except Exception as e:
                    LOGGER.error(f"Failed to write to {SHARED_DATA_FILE}: {e}")

                if stop_button:
                    cap.release()  # Release the capture
                    self.warning_placeholder.empty()  # 종료 시 상태 표시기 지우기
                    self.initialize_shared_data()
                    self.success_placeholder.success("Model loaded successfully!")
                    self.st.stop()

                self.org_frame.image(frame, channels="BGR")  # Display original frame
                self.ann_frame.image(annotated_frame, channels="BGR")  # Display processed frame

if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    inference_instance = Inference(model=model)

    # 서버가 처음 실행될 때 JSON 파일 초기화
    inference_instance.initialize_shared_data()

    # 추론 및 웹 UI 시작
    inference_instance.inference()
