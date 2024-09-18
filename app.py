import streamlit as st
from datetime import datetime
import openai
from typing import Dict, List
import os
from dotenv import load_dotenv
import pinecone

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
    index.upsert(vectors=[(unique_id, embedding, {"text": llm_processed})])
    
    return llm_processed

def query_data(query: str, top_k: int = 5) -> List[Dict]:
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results.matches

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

def query_page():
    st.title("Query Attendance and Leave Data")
    query = st.text_input("Enter your query")
    if st.button("Search"):
        results = query_data(query)
        for i, result in enumerate(results, 1):
            st.subheader(f"Result {i}")
            st.write(result['metadata']['text'])
            st.write(f"Similarity: {result['score']}")

def main():
    st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")
    
    st.sidebar.title("Leave Buddy")
    st.sidebar.image("https://via.placeholder.com/150", use_column_width=True)
    page = st.sidebar.radio("Navigation", ["Attendance", "Leave", "Query"])
    
    if page == "Attendance":
        attendance_page()
    elif page == "Leave":
        leave_page()
    elif page == "Query":
        query_page()

if __name__ == "__main__":
    main()
