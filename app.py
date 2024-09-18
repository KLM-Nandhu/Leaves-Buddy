import streamlit as st
from datetime import datetime
import openai
from typing import Dict
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone with ServerlessSpec
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("leave-buddy-index", serverless_spec=ServerlessSpec())

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# User data
USERS: Dict[str, str] = {
    "Nandhakumar": "nandhakumar@klmsolutions.in",
    "Dhanush": "dhanush@klmsolutions.in",
    "Shubaritha": "shubaritha@klmsolutions.in",
    "Subashree": "subashree@klmsolutions.in",
    "Prateeka": "prateeka@klmsolutions.in",
    "Akshara Shri": "akshara@klmsolutions.in"
}

def get_embedding(text: str) -> list:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def store_data(data: Dict[str, str], data_type: str):
    text = f"{data_type}: {', '.join([f'{k}: {v}' for k, v in data.items()])}"
    embedding = get_embedding(text)
    unique_id = f"{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    index.upsert(vectors=[(unique_id, embedding, {"text": text})])

def attendance_page():
    st.title("Daily Attendance")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox("Select your name", list(USERS.keys()))
    with col2:
        email = USERS[name]
        st.text_input("Email", value=email, disabled=True)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        date = st.date_input("Date")
    with col4:
        time_in = st.time_input("Time In")
    with col5:
        time_out = st.time_input("Time Out")
    
    if st.button("Submit Attendance", key="attend_submit"):
        data = {
            "name": name,
            "email": email,
            "date": str(date),
            "time_in": str(time_in),
            "time_out": str(time_out)
        }
        store_data(data, "attendance")
        st.success("Attendance recorded successfully!")

def leave_page():
    st.title("Leave Application")
    
    col1, col2 = st.columns(2)
    with col1:
        name = st.selectbox("Select your name", list(USERS.keys()))
    with col2:
        email = USERS[name]
        st.text_input("Email", value=email, disabled=True)
    
    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Leave Start Date")
    with col4:
        end_date = st.date_input("Leave End Date")
    
    reason = st.text_area("Reason for Leave")
    
    if st.button("Submit Leave Application", key="leave_submit"):
        data = {
            "name": name,
            "email": email,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "reason": reason
        }
        store_data(data, "leave")
        st.success("Leave application submitted successfully!")

def main():
    st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")
    
    st.sidebar.title("Leave Buddy")
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
    page = st.sidebar.radio("Navigation", ["Attendance", "Leave"])
    
    if page == "Attendance":
        attendance_page()
    elif page == "Leave":
        leave_page()

if __name__ == "__main__":
    main()
