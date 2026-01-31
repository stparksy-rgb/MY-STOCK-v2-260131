import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# 비밀번호 설정
CORRECT_PASSWORD = "6211"

# 비밀번호 확인
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    if st.session_state["password_correct"]:
        return True
    
    st.markdown("""
    <div style='text-align: center; padding: 100px 20px;'>
        <h1 style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5em;'>
        🔐 AI 주식 트레이딩 시스템
        </h1>
        <p style='color: #888; font-size: 1.3em; margin-top: 20px;'>프리미엄 회원 전용</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Access Code", type="password", placeholder="회원 코드 입력", label_visibility="collapsed")
        if st.button("🔓 접속하기", use_container_width=True, type="primary"):
            if password == CORRECT_PASSWORD:
                st.session_state["password_correct"] = True
                st.success("✅ 인증 완료!")
                st.rerun()
            else:
                st.error("❌ 잘못된 접속 코드입니다.")
    
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #666; font-size: 0.9em;'>
        <p>※ 회원 전용 서비스입니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    return False

if not check_password():
    st.stop()

# 페이지 설정
st.set_page_config(layout="wide", page_title="AI 트레이딩 시스템", page_icon="🤖")

# CSS
st.markdown("""
<style>
.stApp { background-color: #000000; color: #e0e0e0; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    height: 60px; padding: 0px 25px; background-color: #1a1a1a;
    border-radius: 10px; color: #ffffff !important;
    font-size: 17px !important; font-weight: bold !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
.metric-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
    border: 2px solid rgba(99, 102, 241, 0.5); border-radius: 15px; padding: 20px; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# 구글 시트에서 종목 불러오기
@st.cache_data(ttl=600)
def load_google_sheet(sheet_url):
    try:
        # URL에서 스프레드시트 ID 추출
        if '/d/' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        else:
            csv_url = sheet_url
        
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"❌ 구글 시트 로딩 실패: {str(e)}")
        return None

# 한국 주식 데이터
@st.cache_data(ttl=300)
def get_data(ticker):
    try:
        clean_ticker = ticker.strip().upper()
        
        if clean_ticker.isdigit() and len(clean_ticker) == 6:
            ticker_symbol = clean_ticker + ".KS"
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="2y")
            
            if df.empty:
                ticker_symbol = clean_ticker + ".KQ"
                stock = yf.Ticker(ticker_symbol)
                df = stock.history(period="2y")
            
            korean_names = {
                '005930': '삼성전자', '000660': 'SK하이닉스', '035720': '카카오',
                '035420': 'NAVER', '005380': '현대차', '000270': '기아',
                '051910': 'LG화학', '006400': '삼성SDI', '207940': '삼성바이오로직스',
                '068270': '셀트리온', '028260': '삼성물산', '042700': '한미반도체',
                '373220': 'LG에너지솔루션', '196170': '알테오젠', '247540': '에코프로비엠'
            }
            
            name = korean_names.get(clean_ticker, f"({clean_ticker})")
        else:
            return None, None
        
        if df.empty:
            return None, None
        
        return df, name
    except:
        return None, None

# 이동평균
def calculate_ma(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    return df

# 스토캐스틱
def calculate_stochastic(df, k_period=8, d_period=5, smooth_k=5):
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%K'] = k.rolling(window=smooth_k).mean()
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df

# RSI
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 매매 신호
def generate_signals(df, oversold=25, overbought=75):
    df['Buy_Signal'] = None
    df['Sell_Signal'] = None
    df['Strong_Buy'] = False
    
    for i in range(1, len(df)):
        if (df['%K'].iloc[i-1] < df['%D'].iloc[i-1] and 
            df['%K'].iloc[i] > df['%D'].iloc[i] and 
            df['%K'].iloc[i] <= oversold and df['%D'].iloc[i] <= oversold):
            df.at[df.index[i], 'Buy_Signal'] = df['Low'].iloc[i] * 0.97
            df.at[df.index[i], 'Strong_Buy'] = True
        elif (df['%K'].iloc[i-1] < df['%D'].iloc[i-1] and 
              df['%K'].iloc[i] > df['%D'].iloc[i] and 
              df['%K'].iloc[i] <= oversold):
            df.at[df.index[i], 'Buy_Signal'] = df['Low'].iloc[i] * 0.97
        elif (df['%K'].iloc[i-1] > df['%D'].iloc[i-1] and 
              df['%K'].iloc[i] < df['%D'].iloc[i] and 
              df['%K'].iloc[i] >= overbought):
            df.at[df.index[i], 'Sell_Signal'] = df['High'].iloc[i] * 1.03
    
    return df

# 백테스팅
def run_backtest(df, initial_capital=10000000):
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(df)):
        if not pd.isna(df['Buy_Signal'].iloc[i]) and position == 0:
            shares = capital // df['Close'].iloc[i]
            if shares > 0:
                position = shares
                buy_price = df['Close'].iloc[i]
                capital -= shares * buy_price
                trades.append({'type': 'buy', 'date': df.index[i], 'price': buy_price, 'shares': shares})
        
        elif not pd.isna(df['Sell_Signal'].iloc[i]) and position > 0:
            sell_price = df['Close'].iloc[i]
            capital += position * sell_price
            profit = (sell_price - buy_price) / buy_price * 100
            trades.append({'type': 'sell', 'date': df.index[i], 'price': sell_price, 'shares': position, 'profit': profit})
            position = 0
    
    if position > 0:
        capital += position * df['Close'].iloc[-1]
    
    total_return = ((capital - initial_capital) / initial_capital) * 100
    sell_trades = [t for t in trades if t['type'] == 'sell']
    
    if sell_trades:
        winning_trades = [t for t in sell_trades if t['profit'] > 0]
        win_rate = len(winning_trades) / len(sell_trades) * 100
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in sell_trades if t['profit'] <= 0]
        avg_loss = abs(np.mean([t['profit'] for t in losing_trades])) if losing_trades else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = 0
        profit_loss_ratio = 0
        winning_trades = []
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': len(sell_trades),
        'winning_trades': len(winning_trades),
        'trades': trades
    }

# 헤더
st.markdown("""
<h1 style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em; margin-bottom: 0;'>
🤖 AI 주식 트레이딩 시스템
</h1>
<p style='text-align: center; color: #888; margin-top: 0;'>인공지능 기반 매매 신호 분석</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([5, 1, 1])
with col3:
    if st.button("🚪 종료", type="secondary"):
        st.session_state["password_correct"] = False
        st.rerun()

# 탭
tab1, tab2, tab3 = st.tabs(["📊 차트 분석", "📈 백테스팅", "💼 포트폴리오"])

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 입력 모드 선택
    input_mode = st.radio(
        "종목 입력 방식",
        ["직접 입력", "구글 시트 테마"],
        horizontal=True
    )
    
    selected_tickers = ""
    
    if input_mode == "직접 입력":
        tickers_input = st.text_area(
            "종목 코드 입력 (6자리)", 
            value="005930, 000660, 035720, 042700", 
            height=120,
            help="예: 005930 (삼성전자), 000660 (SK하이닉스)"
        )
        selected_tickers = tickers_input
    
    else:  # 구글 시트 테마
        st.markdown("#### 📊 구글 시트 연동")
        
        if "sheet_url" not in st.session_state:
            st.session_state.sheet_url = ""
        
        sheet_url = st.text_input(
            "구글 시트 URL",
            value=st.session_state.sheet_url,
            placeholder="https://docs.google.com/spreadsheets/...",
            help="공유 링크를 붙여넣으세요"
        )
        
        if sheet_url:
            st.session_state.sheet_url = sheet_url
            
            with st.spinner("📥 데이터 로딩 중..."):
                df_stocks = load_google_sheet(sheet_url)
            
            if df_stocks is not None and not df_stocks.empty:
                # 컬럼명 확인
                col_theme = None
                col_codes = None
                
                for col in df_stocks.columns:
                    if '테마' in col or '구분' in col:
                        col_theme = col
                    if '코드' in col or 'code' in col.lower():
                        col_codes = col
                
                if col_theme and col_codes:
                    # 테마 목록 추출
                    themes = df_stocks[col_theme].dropna().unique().tolist()
                    
                    st.success(f"✅ {len(df_stocks)}개 데이터, {len(themes)}개 테마 로드 완료!")
                    
                    # 테마 선택 (멀티셀렉트)
                    selected_themes = st.multiselect(
                        "🎯 테마 선택 (여러 개 가능)",
                        themes,
                        default=themes[:2] if len(themes) >= 2 else themes,
                        help="원하는 테마를 선택하세요"
                    )
                    
                    if selected_themes:
                        # 선택된 테마의 종목 필터링
                        filtered_df = df_stocks[df_stocks[col_theme].isin(selected_themes)]
                        
                        # 종목코드 추출 (쉼표로 구분된 경우 처리)
                        all_codes = []
                        for codes_str in filtered_df[col_codes].dropna():
                            codes_str = str(codes_str).strip()
                            # 쉼표로 구분
                            codes_list = [c.strip() for c in codes_str.split(',') if c.strip()]
                            # 공백으로 구분 (쉼표가 없는 경우)
                            if len(codes_list) == 1:
                                codes_list = [c.strip() for c in codes_str.split() if c.strip()]
                            all_codes.extend(codes_list)
                        
                        # 중복 제거
                        unique_codes = list(dict.fromkeys(all_codes))
                        
                        # 6자리 숫자만 필터링
                        valid_codes = [c for c in unique_codes if c.isdigit() and len(c) == 6]
                        
                        selected_tickers = ', '.join(valid_codes)
                        
                        st.caption(f"📌 선택된 종목: {len(valid_codes)}개")
                        
                        # 선택된 테마별 종목 수 표시
                        for theme in selected_themes:
                            theme_codes = filtered_df[filtered_df[col_theme] == theme][col_codes].values
                            if len(theme_codes) > 0:
                                theme_code_list = []
                                for codes_str in theme_codes:
                                    codes_str = str(codes_str).strip()
                                    codes_list = [c.strip() for c in codes_str.split(',') if c.strip()]
                                    if len(codes_list) == 1:
                                        codes_list = [c.strip() for c in codes_str.split() if c.strip()]
                                    theme_code_list.extend(codes_list)
                                valid_theme_codes = [c for c in theme_code_list if c.isdigit() and len(c) == 6]
                                st.caption(f"  • {theme}: {len(valid_theme_codes)}개")
                    else:
                        st.warning("⚠️ 테마를 선택해주세요")
                        selected_tickers = ""
                else:
                    st.error("❌ '테마구분'과 '종목코드' 컬럼을 찾을 수 없습니다")
                    st.info("💡 구글 시트에 '테마구분'과 '종목코드' 컬럼이 있는지 확인하세요")
            else:
                selected_tickers = ""
        else:
            st.info("💡 구글 시트 URL을 입력하세요")
    
    st.markdown("---")
    
    # 지표 설정 추가
    st.subheader("📊 지표 설정")
    col1, col2 = st.columns(2)
    with col1:
        k_period = st.number_input("Fast %K", value=8, min_value=1, max_value=20, step=1)
        oversold = st.slider("매수 기준", 0, 50, 25)
    with col2:
        d_period = st.number_input("Slow %D", value=5, min_value=1, max_value=20, step=1)
        overbought = st.slider("매도 기준", 50, 100, 75)
    
    smooth_k = st.number_input("Smooth %K", value=5, min_value=1, max_value=20, step=1)
    rsi_period = st.number_input("RSI 기간", value=14, min_value=5, max_value=30, step=1)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 15px; border-radius: 10px;'>
        <p style='color: #888; font-size: 0.85em; margin: 0;'>
        ⚠️ <strong>투자 유의사항</strong><br>
        본 서비스는 투자 참고용이며,<br>
        투자 손실 책임은 투자자에게 있습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

# TAB 1: 차트 분석
with tab1:
    if analyze_btn:
        if not selected_tickers:
            st.warning("⚠️ 종목을 입력하거나 구글 시트에서 테마를 선택해주세요")
        else:
            tickers = [t.strip() for t in selected_tickers.split(',') if t.strip()]
            
            st.info(f"🔍 {len(tickers)}개 종목 분석 중...")
            
            for idx, ticker in enumerate(tickers):
                df, name = get_data(ticker)
                
                if df is None or df.empty or len(df) < 60:
                    st.error(f"❌ {ticker}: 데이터를 가져올 수 없습니다.")
                    continue
                
                # 지표 계산
                df = calculate_ma(df)
                df = calculate_stochastic(df, k_period, d_period, smooth_k)
                df = calculate_rsi(df, rsi_period)
                df = generate_signals(df, oversold, overbought)
                
                curr = df.iloc[-1]
                is_strong_buy = curr.get('Strong_Buy', False)
                
                # 종목 정보
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"{name} ({ticker})")
                with col2:
                    price_change = ((curr['Close'] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    st.metric("현재가", f"{curr['Close']:,.0f}원", f"{price_change:+.2f}%")
                
                # 지표 카드
                col1, col2, col3 = st.columns(3)
                with col1:
                    k_color = "#22c55e" if curr['%K'] <= 25 else "#ef4444" if curr['%K'] >= 75 else "#3b82f6"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 14px;'>스토캐스틱</div>
                        <div style='font-size: 32px; font-weight: bold; color: {k_color};'>%K: {curr['%K']:.1f}</div>
                        <div style='color: #aaa; font-size: 16px;'>%D: {curr['%D']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    rsi_color = "#22c55e" if curr['RSI'] <= 30 else "#ef4444" if curr['RSI'] >= 70 else "#a855f7"
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 14px;'>RSI (14)</div>
                        <div style='font-size: 32px; font-weight: bold; color: {rsi_color};'>{curr['RSI']:.1f}</div>
                        <div style='color: #666; font-size: 13px;'>
                            {"과매도" if curr['RSI'] <= 30 else "과매수" if curr['RSI'] >= 70 else "중립"}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 14px;'>AI 신호</div>
                        <div style='font-size: 26px; font-weight: bold; color: {"#22c55e" if is_strong_buy else "#888"};'>
                            {"✅ 적극매수" if is_strong_buy else "⏸️ 대기"}
                        </div>
                        <div style='color: #666; font-size: 12px;'>%K<{oversold} & %D<{oversold} 골든크로스</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 차트
                end_date = df.index[-1]
                start_date = end_date - pd.DateOffset(months=5)
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                  row_heights=[0.65, 0.15, 0.2])
                
                # 캔들차트
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    increasing_line_color='red', decreasing_line_color='blue',
                    name=''), row=1, col=1)
                
                # 이동평균선
                fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], line=dict(color='#FF6B35', width=2),
                                       name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#2979FF', width=3),
                                       name='MA20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='#9D4EDD', width=3),
                                       name='MA60'), row=1, col=1)
                
                # 매매신호
                strong_buy = df[df['Strong_Buy'] == True]
                normal_buy = df[(~df['Buy_Signal'].isna()) & (df['Strong_Buy'] == False)]
                sell = df[~df['Sell_Signal'].isna()]
                
                if len(strong_buy) > 0:
                    fig.add_trace(go.Scatter(
                        x=strong_buy.index, y=strong_buy['Buy_Signal'],
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=25, color='#FF0000',
                                   line=dict(width=2, color='yellow')),
                        text=["적극매수"] * len(strong_buy),
                        textposition="bottom center",
                        textfont=dict(color='#FF0000', size=14),
                        name='적극매수'), row=1, col=1)
                
                if len(normal_buy) > 0:
                    fig.add_trace(go.Scatter(
                        x=normal_buy.index, y=normal_buy['Buy_Signal'],
                        mode='markers+text',
                        marker=dict(symbol='triangle-up', size=15, color='#FF6B35'),
                        text=["매수"] * len(normal_buy),
                        textposition="bottom center",
                        textfont=dict(color='#FF6B35', size=11),
                        name='매수'), row=1, col=1)
                
                if len(sell) > 0:
                    fig.add_trace(go.Scatter(
                        x=sell.index, y=sell['Sell_Signal'],
                        mode='markers+text',
                        marker=dict(symbol='triangle-down', size=18, color='#2979FF'),
                        text=["매도"] * len(sell),
                        textposition="top center",
                        textfont=dict(color='#2979FF', size=13),
                        name='매도'), row=1, col=1)
                
                # 거래량
                colors = ['red' if row['Open'] <= row['Close'] else 'blue' for index, row in df.iterrows()]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors,
                                   name='거래량'), row=2, col=1)
                
                # 스토캐스틱
                fig.add_trace(go.Scatter(x=df.index, y=df['%K'], line=dict(color='#00E5FF', width=2),
                                       name='%K'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['%D'], line=dict(color='#FF6D00', width=2),
                                       name='%D'), row=3, col=1)
                fig.add_hline(y=oversold, line_dash="dash", line_color="#00E676", line_width=2, row=3, col=1)
                fig.add_hline(y=overbought, line_dash="dash", line_color="#FF1744", line_width=2, row=3, col=1)
                
                fig.update_layout(
                    height=700, template="plotly_dark", showlegend=False,
                    hovermode="closest", dragmode='pan',
                    margin=dict(l=50, r=80, t=30, b=40),
                    paper_bgcolor="#000000", plot_bgcolor="#000000",
                    xaxis_rangeslider_visible=False
                )
                
                fig.update_xaxes(
                    showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)',
                    range=[start_date, end_date],
                    tickformat='%Y년 %m월'
                )
                
                fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)',
                               side='right', tickformat=',', ticksuffix='원', row=1, col=1)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)',
                               side='right', row=2, col=1)
                fig.update_yaxes(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)',
                               side='right', range=[0, 100], row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{ticker}_{idx}")
                
                if idx < len(tickers) - 1:
                    st.markdown("---")

# TAB 2: 백테스팅
with tab2:
    if analyze_btn:
        if not selected_tickers:
            st.warning("⚠️ 종목을 입력하거나 구글 시트에서 테마를 선택해주세요")
        else:
            st.subheader("📈 백테스팅 결과")
            tickers = [t.strip() for t in selected_tickers.split(',') if t.strip()]
            
            for ticker in tickers:
                df, name = get_data(ticker)
                if df is None or df.empty or len(df) < 60:
                    continue
                
                df = calculate_ma(df)
                df = calculate_stochastic(df, k_period, d_period, smooth_k)
                df = calculate_rsi(df, rsi_period)
                df = generate_signals(df, oversold, overbought)
                
                results = run_backtest(df)
                
                st.markdown(f"### 📊 {name} ({ticker})")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 13px;'>총 수익률</div>
                        <div style='font-size: 28px; font-weight: bold; color: {"#22c55e" if results['total_return'] > 0 else "#ef4444"};'>
                            {results['total_return']:+.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 13px;'>승률</div>
                        <div style='font-size: 28px; font-weight: bold; color: #3b82f6;'>
                            {results['win_rate']:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 13px;'>총 거래</div>
                        <div style='font-size: 28px; font-weight: bold; color: #a855f7;'>
                            {results['total_trades']}회
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='color: #888; font-size: 13px;'>손익비</div>
                        <div style='font-size: 28px; font-weight: bold; color: #f59e0b;'>
                            {results['profit_loss_ratio']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")

# TAB 3: 포트폴리오
with tab3:
    st.subheader("💼 포트폴리오 관리")
    st.info("🚧 포트폴리오 기능은 업데이트 예정입니다")
