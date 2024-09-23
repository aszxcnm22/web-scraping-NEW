import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from collections import defaultdict
from datetime import datetime

# ฟังก์ชันทำให้ชื่อคอลัมน์ไม่ซ้ำกัน
def make_unique_columns(columns):
    counts = defaultdict(int)
    unique_columns = []
    for col in columns:
        if counts[col]:
            unique_columns.append(f"{col}_{counts[col]}")
        else:
            unique_columns.append(col)
        counts[col] += 1
    return unique_columns

# โหลดโมเดลที่บันทึกไว้
model_filename = 'linear_regression_model.pkl'
try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    st.error(f"ไม่พบไฟล์โมเดล `{model_filename}` โปรดตรวจสอบว่าไฟล์มีอยู่ในไดเรกทอรีที่ถูกต้อง")
    st.stop()

# ฟังก์ชันในการทำนายราคาหุ้น
def predict_stock_price(input_data):
    df = pd.DataFrame(input_data)
    required_columns = ['Open', 'High', 'Low', 'Volume', 'Open_lag', 'High_lag', 'Low_lag', 'Volume_lag']
    if not all(column in df.columns for column in required_columns):
        st.error(f"ข้อมูลต้องมีคอลัมน์ดังนี้: {required_columns}")
        return None
    predictions = model.predict(df[required_columns])
    return predictions

# ส่วนของการแสดงผล Streamlit
st.title("การทำนายราคาหุ้น")
st.sidebar.header("การตั้งค่า")

# ผู้ใช้อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ที่มีข้อมูลราคาหุ้น", type=['csv'])

if uploaded_file is not None:
    try:
        input_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ CSV ได้: {e}")
        st.stop()
    
    st.write("ข้อมูลที่อัปโหลด:")
    st.write(input_data)
    
    # ลบช่องว่างและอักขระพิเศษในชื่อคอลัมน์
    input_data.columns = input_data.columns.str.strip().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    
    # ทำให้ชื่อคอลัมน์ไม่ซ้ำกัน
    input_data.columns = make_unique_columns(input_data.columns)
    
    st.write("ชื่อคอลัมน์ในไฟล์ที่อัปโหลด (ไม่ซ้ำกัน):")
    st.write(input_data.columns)
    
    # ตรวจสอบว่ามีคอลัมน์ Date
    if 'Date' not in input_data.columns:
        st.error("ไฟล์ CSV ต้องมีคอลัมน์ 'Date'")
    else:
        # แปลงคอลัมน์ Date เป็น datetime
        try:
            input_data['Date'] = pd.to_datetime(input_data['Date'])
        except Exception as e:
            st.error(f"ไม่สามารถแปลงคอลัมน์ 'Date' เป็น datetime ได้: {e}")
            st.stop()
        
        # เรียงลำดับข้อมูลตามวันที่
        input_data.sort_values('Date', inplace=True)
        input_data.reset_index(drop=True, inplace=True)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['Open', 'High', 'Low', 'Volume', 'Close']
        if not all(column in input_data.columns for column in required_columns):
            st.error(f"กรุณาอัปโหลดไฟล์ที่มีคอลัมน์ดังนี้: {', '.join(required_columns)}")
        else:
            # สร้างฟีเจอร์ lag
            input_data['Open_lag'] = input_data['Open'].shift(1)
            input_data['High_lag'] = input_data['High'].shift(1)
            input_data['Low_lag'] = input_data['Low'].shift(1)
            input_data['Volume_lag'] = input_data['Volume'].shift(1)
            input_data.dropna(inplace=True)
            
            st.write("ข้อมูลหลังจากสร้างฟีเจอร์ lag:")
            st.write(input_data.head())
    
            # ทำนายราคาหุ้น
            st.write("ราคาหุ้นที่คาดการณ์ได้:")
            predictions = predict_stock_price(input_data)
            if predictions is not None:
                input_data['Predicted_Close'] = predictions
                
                # เพิ่มตัวเลือกใน sidebar สำหรับการเลือกช่วงวันที่
                st.sidebar.subheader("เลือกช่วงวันที่ที่จะแสดง")
                
                min_date = input_data['Date'].min().date()
                max_date = input_data['Date'].max().date()
                
                start_date, end_date = st.sidebar.date_input(
                    "เลือกช่วงวันที่",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                
                # ตรวจสอบว่าผู้ใช้เลือกช่วงวันที่ที่ถูกต้อง
                if start_date > end_date:
                    st.error("วันที่เริ่มต้นต้องก่อนหรือเท่ากับวันที่สิ้นสุด")
                else:
                    # กรองข้อมูลตามช่วงวันที่ที่เลือก
                    mask = (input_data['Date'].dt.date >= start_date) & (input_data['Date'].dt.date <= end_date)
                    displayed_data = input_data.loc[mask, ['Date', 'Close', 'Predicted_Close']]
                    
                    if displayed_data.empty:
                        st.warning("ไม่มีข้อมูลในช่วงวันที่ที่เลือก")
                    else:
                        st.write(f"ข้อมูลที่จะแสดงระหว่างวันที่ {start_date} ถึง {end_date}:")
                        st.write(displayed_data)
            
                        # เพิ่มตัวเลือกในการเลือกประเภทของกราฟที่จะแสดง
                        plot_options = st.sidebar.multiselect(
                            "เลือกข้อมูลที่จะแสดงในกราฟ",
                            options=['Close', 'Predicted_Close'],
                            default=['Close', 'Predicted_Close']
                        )
            
                        # ตรวจสอบว่า 'Date' มีข้อมูลหรือไม่
                        if displayed_data['Date'].isnull().any():
                            st.error("คอลัมน์ 'Date' มีค่าว่างอยู่")
                        else:
                            # สร้างกราฟเปรียบเทียบราคาจริงและราคาที่คาดการณ์ด้วย Plotly
                            fig = go.Figure()
                            if 'Close' in plot_options:
                                fig.add_trace(go.Scatter(
                                    x=displayed_data['Date'],
                                    y=displayed_data['Close'],
                                    mode='lines',
                                    name='Actual Close Price',
                                    line=dict(color='blue')
                                ))
                            if 'Predicted_Close' in plot_options:
                                fig.add_trace(go.Scatter(
                                    x=displayed_data['Date'],
                                    y=displayed_data['Predicted_Close'],
                                    mode='lines',
                                    name='Predicted Close Price',
                                    line=dict(color='red', dash='dash')
                                ))
                            fig.update_layout(
                                title='Actual vs Predicted Stock Prices',
                                xaxis_title='Date',
                                yaxis_title='Stock Price',
                                xaxis=dict(rangeslider=dict(visible=True), type='date'),
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            