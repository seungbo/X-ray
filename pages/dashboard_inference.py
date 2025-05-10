import streamlit as st
import time
import pandas as pd
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
    <h1 style='color:#FF8000; font-size:40px; margin-top:10px; font-family: \"Archivo\", sans-serif;'>
        X-RayVision 실시간 대시보드
    </h1>
</div>
""", unsafe_allow_html=True)

# 로그 표시 영역
st.subheader("모델 예측 로그 (최근 10개)")
log_container = st.empty()

# 차트 영역 컨테이너
chart_container = st.empty()

# 데이터 읽기 함수
def read_shared_data():
    shared_data = DEFAULT_SHARED_DATA.copy()
    try:
        with open(SHARED_DATA_FILE, 'r') as f:
            shared_data = json.load(f)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        st.warning(f"{SHARED_DATA_FILE} 파일을 읽는 중 오류가 발생했습니다. 파일 내용을 확인해주세요.")
    except Exception as e:
        st.error(f"데이터 로딩 중 예상치 못한 오류 발생: {e}")
    return shared_data

# 차트 그리기 함수
def update_charts(current_counts, cumulative_counts):
    import altair as alt
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("현재 감지된 객체")
        if not current_counts:
            current_counts = DEFAULT_SHARED_DATA['current_counts']
        current_df = pd.DataFrame({
            '카테고리': list(current_counts.keys()),
            'count': list(current_counts.values())
        })
        if current_df.empty:
            current_df = pd.DataFrame({
                '카테고리': list(DEFAULT_SHARED_DATA['current_counts'].keys()),
                'count': list(DEFAULT_SHARED_DATA['current_counts'].values())
            })
        chart_current = alt.Chart(current_df).mark_bar().encode(
            x=alt.X('카테고리:N', axis=alt.Axis(labelAngle=0, title=None, labelFontSize=18)),
            y=alt.Y('count:Q', axis=alt.Axis(title=None, labelFontSize=18))
        ).properties(height=300)
        st.altair_chart(chart_current, use_container_width=True)
    with col2:
        st.subheader("객체 누적 카운트")
        if not cumulative_counts:
            cumulative_counts = DEFAULT_SHARED_DATA['cumulative_counts']
        cumulative_df = pd.DataFrame({
            '카테고리': list(cumulative_counts.keys()),
            'count': list(cumulative_counts.values())
        })
        chart_cumulative = alt.Chart(cumulative_df).mark_bar().encode(
            x=alt.X('카테고리:N', axis=alt.Axis(labelAngle=0, title=None, labelFontSize=18)),
            y=alt.Y('count:Q', axis=alt.Axis(title=None, labelFontSize=18))
        ).properties(height=300)
        st.altair_chart(chart_cumulative, use_container_width=True)

# 실시간 갱신 루프 (깜빡임 최소화)
REFRESH_INTERVAL_SECONDS = 0.5  # 0.5초마다 갱신
while True:
    shared_data = read_shared_data()
    current_counts_data = shared_data.get('current_counts', DEFAULT_SHARED_DATA['current_counts'])
    cumulative_counts_data = shared_data.get('cumulative_counts', DEFAULT_SHARED_DATA['cumulative_counts'])
    log_messages_data = shared_data.get('log_messages', DEFAULT_SHARED_DATA['log_messages'])

    # 로그 메시지 표시 (st.code로 출력, 최근 10개만)
    with log_container:
        if log_messages_data:
            st.code("\n".join(log_messages_data[-10:]), language=None)
        else:
            st.info("아직 로그 메시지가 없습니다. 감지가 시작되면 여기에 표시됩니다.")

    # 차트 표시
    with chart_container:
        update_charts(current_counts_data, cumulative_counts_data)

    time.sleep(REFRESH_INTERVAL_SECONDS)
    st.rerun()
