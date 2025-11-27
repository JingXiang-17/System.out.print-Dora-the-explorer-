import streamlit as st
import pandas as pd
from optimization import optimize_flight_delays

st.title("Flight Delay Optimization Demo ✈️")

# 上传 CSV
uploaded_file = st.file_uploader("Upload CSV with flight predictions", type="csv")

# 默认示例数据
if uploaded_file is None:
    st.info("No CSV uploaded. Using default example data.")
    data = {
        "flight_id": ["DL101", "DL102", "DL103"],
        "predicted_delay": [22.3, -1.5, 40.0],
        "risk": [True, False, True]
    }
    df = pd.DataFrame(data)
else:
    df = pd.read_csv(uploaded_file)

st.subheader("Raw Data")
st.dataframe(df)

# 转换数据格式
flights = []
for idx, row in df.iterrows():
    flights.append({
        "flight": row["flight_id"],
        "predicted_delay": row["predicted_delay"],
        "risk": row["risk"]
    })

# 调用优化函数
results = optimize_flight_delays(flights)

st.subheader("Optimization Results")

# 高亮显示每个航班
for r in results:
    flight = r["flight"]
    predicted_delay = r["predicted_delay"]
    delay_flag = "Delayed" if r["delay_flag"] else ("Early" if r["early_flag"] else "On Time")
    risk_flag = "High Risk" if r["high_risk_flag"] else "Normal"

    # 根据状态设置颜色 (必须缩进在循环里)
    if delay_flag == "Delayed":
        color = "#FFB6B6"  # 红色
    elif risk_flag == "High Risk":
        color = "#FFD580"  # 橙色
    elif delay_flag == "Early":
        color = "#B6E3FF"  # 蓝色
    elif delay_flag == "On Time":
        color = "#A0E7E5"  # 青色
    else:
        color = "#FFFFFF"  # 默认白色

    st.markdown(f"<div style='background-color: {color}; padding:10px; border-radius:5px'>", unsafe_allow_html=True)
    st.markdown(f"**Flight:** {flight} | **Predicted Delay (min):** {predicted_delay} | **Status:** {delay_flag} | **Risk:** {risk_flag}")

    # 每条 recommendation 单独显示
    st.write("**Recommendations:**")
    for rec in r["recommendations"]:
        st.markdown(f"- {rec}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
