import streamlit as st
from datetime import datetime, timedelta
import openai
from typing import Dict, List
import os
from dotenv import load_dotenv
import pinecone
import pandas as pd
import io

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone.Index("leave-buddy-index")

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

def check_pinecone_connection() -> bool:
    try:
        index.describe_index_stats()
        return True
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {str(e)}")
        return False

def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def process_with_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant processing attendance and leave data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

def store_data(data: Dict[str, str], data_type: str):
    text = f"{data_type}: {', '.join([f'{k}: {v}' for k, v in data.items()])}"
    
    # Process the data with LLM
    llm_processed = process_with_llm(f"Process this {data_type} data and provide a summary: {text}")
    
    # Get embedding for the LLM-processed text
    embedding = get_embedding(llm_processed)
    
    unique_id = f"{data_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    index.upsert(vectors=[(unique_id, embedding, {"text": llm_processed, "original_data": data, "type": data_type})])
    
    return llm_processed

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
        llm_response = store_data(data, "attendance")
        st.success("Attendance recorded successfully!")
        st.info(f"LLM Processing Result: {llm_response}")

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
        llm_response = store_data(data, "leave")
        st.success("Leave application submitted successfully!")
        st.info(f"LLM Processing Result: {llm_response}")

def download_page():
    st.title("Download Attendance and Leave Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        from_date = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
    with col2:
        to_date = st.date_input("To Date", value=datetime.now())
    with col3:
        staff_options = ["All"] + list(USERS.keys())
        selected_staff = st.selectbox("Select Staff", staff_options)

    if st.button("Download Excel"):
        # Query Pinecone for data within the date range
        query_embedding = get_embedding(f"Data from {from_date} to {to_date}")
        results = index.query(vector=query_embedding, top_k=10000, include_metadata=True)

        # Process and filter the results
        data = []
        for match in results.matches:
            original_data = match.metadata.get('original_data', {})
            record_date = original_data.get('date') or original_data.get('start_date')
            if record_date:
                record_date = datetime.strptime(record_date, "%Y-%m-%d").date()
                if from_date <= record_date <= to_date:
                    if selected_staff == "All" or original_data.get('name') == selected_staff:
                        data.append(original_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Create Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')

        # Offer download
        st.download_button(
            label="Download Excel file",
            data=output.getvalue(),
            file_name=f"attendance_leave_data_{from_date}_to_{to_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Store download record in Pinecone
        download_data = {
            "action": "download",
            "from_date": str(from_date),
            "to_date": str(to_date),
            "staff": selected_staff,
            "timestamp": str(datetime.now())
        }
        store_data(download_data, "download_record")

def main():
    st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")
    
    if not check_pinecone_connection():
        st.error("Failed to connect to Pinecone. Please check your API key and try again.")
        return

    st.sidebar.title("Leave Buddy")
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
    page = st.sidebar.radio("Navigation", ["Attendance", "Leave", "Download"])
    
    if page == "Attendance":
        attendance_page()
    elif page == "Leave":
        leave_page()
    elif page == "Download":
        download_page()

if __name__ == "__main__":
    main()
