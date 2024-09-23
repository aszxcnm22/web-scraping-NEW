import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# โหลดโมเดลที่บันทึกไว้
model_filename = 'linear_regression_model.pkl'
model = joblib.load(model_filename)

# ฟังก์ชันในการทำนายราคาหุ้น
def predict_stock_price(input_data):
    # สร้าง DataFrame จากข้อมูลที่ผู้ใช้อัปโหลด
    df = pd.DataFrame(input_data)

    # ตรวจสอบว่ามีคอลัมน์ที่ต้องใช้ในโมเดลหรือไม่
    required_columns = ['Open', 'High', 'Low', 'Volume()', 'Open_lag', 'High_lag', 'Low_lag', 'Volume_lag']
    if not all(column in df.columns for column in required_columns):
        st.error(f"ข้อมูลต้องมีคอลัมน์ดังนี้: {required_columns}")
        return None

    # ใช้โมเดลในการทำนายราคาหุ้น
    predictions = model.predict(df[required_columns])
    return predictions

# ส่วนของการแสดงผล Streamlit
st.title("Stock Price Prediction")

# ผู้ใช้อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV ที่มีข้อมูลราคาหุ้น", type=['csv'])

if uploaded_file is not None:
    # อ่านข้อมูลจากไฟล์ CSV
    input_data = pd.read_csv(uploaded_file)

    # แสดงข้อมูลที่ผู้ใช้อัปโหลด
    st.write("ข้อมูลที่อัปโหลด:")
    st.write(input_data)

    # แสดงชื่อคอลัมน์เพื่อการตรวจสอบ
    st.write("ชื่อคอลัมน์ในไฟล์ที่อัปโหลด:")
    st.write(input_data.columns)

    # ลบช่องว่างที่อาจมีในชื่อคอลัมน์
    input_data.columns = input_data.columns.str.strip()

    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_columns = ['Open', 'High', 'Low', 'Volume()']
    if not all(column in input_data.columns for column in required_columns):
        st.error(f"ข้อมูลต้องมีคอลัมน์ดังนี้: {required_columns}")
    else:
        # สร้างฟีเจอร์ lag
        input_data['Open_lag'] = input_data['Open'].shift(1)
        input_data['High_lag'] = input_data['High'].shift(1)
        input_data['Low_lag'] = input_data['Low'].shift(1)
        input_data['Volume_lag'] = input_data['Volume()'].shift(1)
        input_data.dropna(inplace=True)

        # ทำนายราคาหุ้น
        st.write("ราคาหุ้นที่คาดการณ์ได้:")
        predictions = predict_stock_price(input_data)
        if predictions is not None:
            input_data['Predicted_Close'] = predictions
            st.write(input_data[['Date', 'Close', 'Predicted_Close']])

            # สร้างกราฟเปรียบเทียบราคาจริงและราคาที่คาดการณ์
            st.write("กราฟเปรียบเทียบราคาจริงและราคาที่คาดการณ์:")
            plt.figure(figsize=(20, 15))
            plt.plot(input_data['Date'], input_data['Close'], label='Actual Close Price', color='blue')
            plt.plot(input_data['Date'], input_data['Predicted_Close'], label='Predicted Close Price', color='red', linestyle='--')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.title('Actual vs Predicted Stock Prices')
            plt.legend()
            st.pyplot(plt)
