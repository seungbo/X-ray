import streamlit as st
import time
import pandas as pd
# import altair as alt
import json

# ① 페이지 설정 (반드시 최상단에!)
st.set_page_config(page_title="X-RayVision 대시보드", layout="wide")

# 공유 데이터 파일 경로 정의 (main_inference.py와 동일해야 함)
SHARED_DATA_FILE = "shared_data.json"

# 기본 데이터 구조 (파일 없을 경우 대비)
DEFAULT_SHARED_DATA = {'current_counts': {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}, 'cumulative_counts': {'일반물품': 0, '위해물품': 0, '정보저장매체': 0}, 'log_messages': []}


# 제목 설정
st.markdown("""
<div style='text-align: center;'>
    <h1 style='color:#FF8000; font-size:40px; margin-top:10px; font-family: "Archivo", sans-serif;'>
        X-RayVision 실시간 대시보드
    </h1>
</div>
""", unsafe_allow_html=True)

# 로그 표시 영역
st.subheader("모델 예측 로그 (최근 10개)")
log_container = st.container()

# 통계 차트를 그리는 함수 정의
# def update_charts():
def update_charts(current_counts, cumulative_counts):
    """주어진 데이터를 사용하여 차트를 업데이트합니다."""
    import altair as alt # 함수 내에서 altair 임포트 (st.rerun() 시 필요할 수 있음)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("현재 감지된 객체")
        # 데이터가 없거나 비어있는 경우 기본값 처리
        if not current_counts:
            current_counts = DEFAULT_SHARED_DATA['current_counts']

        current_df = pd.DataFrame({
            # '카테고리': list(st.session_state['current_counts'].keys()),
            # 'count': list(st.session_state['current_counts'].values())
            '카테고리': list(current_counts.keys()),
            'count': list(current_counts.values())
        })
        # 데이터프레임이 비어있으면 빈 차트 대신 기본 차트 표시
        if current_df.empty:
             current_df = pd.DataFrame({
                '카테고리': list(DEFAULT_SHARED_DATA['current_counts'].keys()),
                'count': list(DEFAULT_SHARED_DATA['current_counts'].values())
            })

        chart_current = alt.Chart(current_df).mark_bar().encode(
            # x축에 제목 제거 및 라벨 회전 각도 0도로 설정
            x=alt.X('카테고리:N', axis=alt.Axis(labelAngle=0, title=None)),
            # y축에 제목 제거
            y=alt.Y('count:Q', axis=alt.Axis(title=None))
        ).properties(height=300)
        st.altair_chart(chart_current, use_container_width=True)
    
    with col2:
        st.subheader("객체 누적 카운트")
        # 데이터가 없거나 비어있는 경우 기본값 처리
        if not cumulative_counts:
            cumulative_counts = DEFAULT_SHARED_DATA['cumulative_counts']

        cumulative_df = pd.DataFrame({
            # '카테고리': list(st.session_state['cumulative_counts'].keys()),
            # 'count': list(st.session_state['cumulative_counts'].values())
            '카테고리': list(cumulative_counts.keys()),
            'count': list(cumulative_counts.values())
        })
        chart_cumulative = alt.Chart(cumulative_df).mark_bar().encode(
            # x축에 제목 제거 및 라벨 회전 각도 0도로 설정
            x=alt.X('카테고리:N', axis=alt.Axis(labelAngle=0, title=None)),
            # y축에 제목 제거
            y=alt.Y('count:Q', axis=alt.Axis(title=None))
        ).properties(height=300)
        st.altair_chart(chart_cumulative, use_container_width=True)
        
REFRESH_INTERVAL_SECONDS = 3 # 갱신 간격 (초)

while True:
    # 공유 파일에서 데이터 읽기 시도
    shared_data = DEFAULT_SHARED_DATA.copy() # 기본값으로 시작
    try:
        with open(SHARED_DATA_FILE, 'r') as f:
            shared_data = json.load(f)
    except FileNotFoundError:
        # 파일이 아직 생성되지 않았을 수 있음 (메인 앱 시작 전)
        pass # 기본값 사용
    except json.JSONDecodeError:
        # 파일이 비어있거나 손상되었을 수 있음
        st.warning(f"{SHARED_DATA_FILE} 파일을 읽는 중 오류가 발생했습니다. 파일 내용을 확인해주세요.")
        # 기본값 사용 또는 이전 상태 유지 (여기서는 기본값 사용)
    except Exception as e:
        st.error(f"데이터 로딩 중 예상치 못한 오류 발생: {e}")
        # 기본값 사용

    current_counts_data = shared_data.get('current_counts', DEFAULT_SHARED_DATA['current_counts'])
    cumulative_counts_data = shared_data.get('cumulative_counts', DEFAULT_SHARED_DATA['cumulative_counts'])
    log_messages_data = shared_data.get('log_messages', DEFAULT_SHARED_DATA['log_messages'])
    

    # 로그 메시지 표시 영역 업데이트
    with log_container:
        if log_messages_data:
            log_text_area = st.text_area("탐지 로그",
                                        "\n".join(log_messages_data),
                                        height=200,
                                        key=f"log_area_{time.time()}") # key를 변경하여 강제 리렌더링
        else:
            st.info("아직 로그 메시지가 없습니다. 감지가 시작되면 여기에 표시됩니다.")

    # 차트 업데이트 (읽어온 데이터 전달)
    update_charts(current_counts_data, cumulative_counts_data)

    # 지정된 시간만큼 대기
    time.sleep(REFRESH_INTERVAL_SECONDS)

    # 페이지 강제 새로고침 (st.rerun 사용)
    st.rerun()
