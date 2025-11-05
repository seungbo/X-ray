# SW-AI: 공항 보안 검색대 위험물품 자동 탐지 시스템

>
> YOLO 기반 객체 인식·인스턴스 세그멘테이션으로 공항 검색 효율과 안전성을 동시에 강화합니다.

---

## 프로젝트 개요
공항 보안 검색 과정은 숙련된 인력의 육안 판독에 의존하기 때문에 **인적 오류, 처리 지연, 피로 누적** 등의 문제가 상존합니다. SW-AI 프로젝트를 통해 X‑ray 이미지를 딥러닝 모델에 입력하여 **위해물품·일반물품·정보저장매체를 실시간으로 탐지**함으로써 이러한 문제를 해소하고자 합니다.  

### 주요 목표
- 다양한 각도·복잡한 배경에서도 높은 탐지 정확도 달성  
- 객체 인식(Object Detection)과 인스턴스 세그멘테이션(Instance Segmentation)을 비교·분석 
- 하이퍼파라미터 튜닝·데이터 증강을 통한 성능 최적화  

---

## 데이터셋
- **출처**: 공항 보안 검색대 실제 장비로 촬영한 X‑ray 이미지 데이터셋(AI-Hub)  
- **구성**: 컬러·흑백 이미지, COCO 형식의 라벨(JSON)
- **크기**: 817GB의 수십만 장 규모 → 용량 문제로 컬러 이미지만 사용  
- **분할 비율**: Train 80 / Val 10 / Test 10 %  
- **전처리**: 클래스 재정의, 좌표 정규화, 데이터 증강(RandomFlip, Mosaic 등)  

---

## 모델 아키텍처
 

---

## 실험 설정
```python
# 1단계 학습
# 핵심 하이퍼파라미터
train_results = model.train(
    project="X-ray",
    name="X-ray-V1",
    data="data.yaml",
    patience=10,
    epochs=20,
    warmup_epochs=3,
    save_period=1,
    lr0=0.001, lrf=0.1,
    optimizer="AdamW",
    imgsz=768,
    device=0,
    batch=32,
    workers=12,
    close_mosaic=4,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.2,
    degrees=5.0, copy_paste=0.2,
    mosaic=0.8,
    auto_augment=None,
    erasing=0.0,
    freeze=12,
    cos_lr=True,
    mask_ratio=2,
    plots=True,
)
```

```python
# 2단계 학습
# 핵심 하이퍼파라미터
train_results = model.train(
    project="X-ray",
    name="X-ray-V2",
    data="data.yaml",
    epochs=100,
    patience=30,
    warmup_epochs=5,
    lr0=8e-5, lrf=0.02,
    weight_decay=5e-4,
    optimizer="AdamW",
    cos_lr=True,
    batch=8,
    workers=12,
    device=0,
    mosaic=0.4,
    close_mosaic=20,
    multi_scale=True,
    hsv_h=0.005, hsv_s=0.3, hsv_v=0.2,
    degrees=5.0, copy_paste=0.2,
    auto_augment=None,
    erasing=0.0,
    freeze=None,
    overlap_mask=False,
    mask_ratio=2,
    save_period=5,
    plots=True,
    seed=42,
    deterministic=False,
)
```


- **모니터링**: Weights & Biases(W&B)로 실시간 모니터링  

---

## 결과

<img width="1178" alt="스크린샷 2025-04-14 오전 9 51 56" src="https://github.com/user-attachments/assets/bea7ff8d-c8c8-49aa-89b0-a37e8b5eda7e" />

- **Instance Segmentation > Object Detection**: 복잡한 배경·겹침 상황에서 더 안정적인 경계 추정  
- 클래스 불균형으로 특정 클래스(예: 배터리 팩)에서 오차↑ → 언더샘플링·Focal Loss로 개선 가능  

---

## 실행 방법

```bash
1. 소스 가져오기
git clone https://github.com/seungbo/X-ray
cd X-ray

2. 가상환경 설정
[맥]
python3 -m venv venv
source venv/bin/activate

[윈도우(cmd)]
python -m venv venv
venv\Scripts\activate

3. 라이브러리 설치
[맥]
pip install --upgrade pip
pip install streamlit
pip install ultralytics

[윈도우(cmd)]
python -m pip install --upgrade pip
python -m pip install streamlit
python -m pip install ultralytics
# GPU 사용 시 아래 명령어 사용 - cuda 11.8 -> 호환성 확인 필요
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118


4. 애플리케이션 실행
[맥]
streamlit run main_inference.py

[윈도우(cmd)]
python -m streamlit run main_inference.py

5. 가상환경 비활성화
deactivate
```
---
