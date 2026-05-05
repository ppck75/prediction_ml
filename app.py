import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Matplotlib Setting
# =========================
# 그래프 안의 한글을 영어로 바꾸었기 때문에 한글 폰트 설정은 제거했습니다.
# 마이너스 기호 깨짐만 방지합니다.
plt.rcParams['axes.unicode_minus'] = False

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="농산물 가격 예측 대시보드",
    page_icon="🥬",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background-color: #F7F8F1;
}

/* 메인 콘텐츠 여백 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #EEF3E6;
    border-right: 1px solid #DDE6D2;
}

/* 사이드바 내부 여백 */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 2rem;
}

/* 메인 타이틀 */
h1 {
    color: #2F3E2F;
    font-weight: 800;
    letter-spacing: -0.5px;
    padding-bottom: 0.4rem;
}

/* 서브헤더 */
h2, h3 {
    color: #3F5F3F;
    font-weight: 700;
}

/* 일반 텍스트 */
p, label, span {
    color: #333333;
}

/* 안내 박스 */
.intro-box {
    background-color: #FFFFFF;
    padding: 1.1rem 1.3rem;
    border-radius: 16px;
    border-left: 6px solid #6F9E65;
    box-shadow: 0 4px 14px rgba(76, 104, 62, 0.08);
    margin-bottom: 1.5rem;
    line-height: 1.7;
    color: #3A3A3A;
}

/* 사이드바 안내 박스 */
.sidebar-guide {
    background-color: #FFFFFF;
    padding: 1rem 1.1rem;
    border-radius: 15px;
    border: 1px solid #DDE6D2;
    box-shadow: 0 3px 10px rgba(76, 104, 62, 0.06);
    font-size: 14px;
    line-height: 1.75;
    color: #333333;
}

/* 카드형 섹션 느낌 */
div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stPyplot"] {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(76, 104, 62, 0.08);
}

/* 버튼, 체크박스 주변 톤 */
.stCheckbox {
    background-color: transparent;
}

/* 입력 위젯 라운드 느낌 */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 12px;
}

/* 구분선 */
hr {
    border: none;
    border-top: 1px solid #DDE6D2;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# =========================
# Data Load
# =========================
csv_file_path = 'data/streamlit_data.csv'

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

df = load_data(csv_file_path)

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
else:
    st.error("Date column not found in the CSV file.")


# =========================
# Data Preprocessing
# =========================
def preprocess_data(df):
    cutoff_date = pd.to_datetime('2020-09-28')
    cols_to_nan = ['cabbage', 'radish', 'garlic', 'onion', 'daikon', 'cilantro', 'artichoke']

    # 실제 데이터에 존재하는 컬럼만 처리하도록 안전장치 추가
    cols_to_nan = [col for col in cols_to_nan if col in df.columns]

    df.loc[df.index > cutoff_date, cols_to_nan] = np.nan
    return df


# =========================
# Plot Function
# =========================
def plot_predictions_over_time(df, vegetables, rolling_mean_window):
    fig, ax = plt.subplots(figsize=(14, 7))

    # 내추럴 톤 컬러 팔레트
    colors = [
        '#5B8C5A',  # soft green
        '#D9A441',  # warm yellow
        '#7A9E7E',  # muted green
        '#B47B48',  # brown orange
        '#8FA86E',  # olive
        '#C9A66B',  # wheat
        '#6B8F71'   # deep sage
    ]

    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    for i, veg in enumerate(vegetables):
        color = colors[i % len(colors)]

        ax.plot(
            df.index,
            df[veg],
            label=veg,
            linewidth=2.5,
            color=color
        )

        rolling_mean = df[veg].rolling(window=rolling_mean_window).mean()

        ax.plot(
            df.index,
            rolling_mean,
            label=f'{veg} ({rolling_mean_window}-day Rolling Mean)',
            linestyle='--',
            linewidth=2,
            alpha=0.75,
            color=color
        )

    ax.set_title(
        'Price Trend by Product',
        fontsize=17,
        fontweight='bold',
        color='#2F3E2F',
        pad=16
    )

    ax.set_xlabel('Date', fontsize=12, color='#444444', labelpad=10)
    ax.set_ylabel('Price', fontsize=12, color='#444444', labelpad=10)

    ax.grid(
        True,
        color='#DDE3D5',
        linestyle='--',
        linewidth=0.8,
        alpha=0.9
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B8C2B0')
    ax.spines['bottom'].set_color('#B8C2B0')

    ax.tick_params(axis='x', colors='#555555', labelsize=10)
    ax.tick_params(axis='y', colors='#555555', labelsize=10)

    ax.legend(
        fontsize=10,
        frameon=False,
        ncol=2,
        loc='best'
    )

    fig.tight_layout()
    st.pyplot(fig)


# =========================
# Main Data
# =========================
df = preprocess_data(df)

metric_file_path = 'data/metric_summary.csv'
metric_summary = pd.read_csv(metric_file_path)
metric_summary.set_index('product', inplace=True)


# =========================
# Header
# =========================
st.title('🥬 농산물 가격 예측 대시보드')

st.markdown("""
<div class="intro-box">
선택한 <b>조회 기간</b>과 <b>품목</b>에 따라 농산물 가격 추이와 이동평균선을 확인할 수 있습니다.<br>
아래에서는 품목별 시계열 흐름과 예측 성능 요약표를 함께 제공합니다.
</div>
""", unsafe_allow_html=True)


# =========================
# Sidebar
# =========================
st.sidebar.markdown("## 📅 조회 기간")
start_date = st.sidebar.date_input('시작일', df.index.min())
end_date = st.sidebar.date_input('마지막일', df.index.max())

st.sidebar.markdown("---")

st.sidebar.markdown("## 🥕 품목 선택")
sorted_vegetables = sorted(df.columns)
vegetables = st.sidebar.multiselect('조회 품목:', sorted_vegetables)

st.sidebar.markdown("## 📈 이동평균 설정")
rolling_mean_window = st.sidebar.slider(
    'Rolling Mean Window',
    min_value=1,
    max_value=30,
    value=7
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class="sidebar-guide">
<b>품목명 안내</b><br><br>
배추 → cabbage<br>
무 → radish<br>
마늘 → garlic<br>
양파 → onion<br>
대파 → daikon<br>
건고추 → cilantro<br>
깻잎 → artichoke
</div>
""", unsafe_allow_html=True)


# =========================
# Filtering
# =========================
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered_df = df.loc[start_date:end_date]


# =========================
# Main Chart Section
# =========================
st.subheader('품목별 예측 대시보드')

if vegetables:
    plot_predictions_over_time(filtered_df, vegetables, rolling_mean_window)
else:
    st.info('왼쪽 사이드바에서 하나 이상의 품목을 선택해주세요.')


# =========================
# Filtered DataFrame
# =========================
st.markdown("---")

if st.checkbox('Show Filtered DataFrame'):
    st.caption('선택한 기간에 해당하는 원본 데이터입니다.')
    st.dataframe(filtered_df, use_container_width=True)


# =========================
# Metric Summary
# =========================
st.subheader('정확도 Summary')
st.caption('품목별 예측 성능 지표를 요약한 표입니다.')

st.dataframe(
    metric_summary,
    use_container_width=True
)