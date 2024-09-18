import streamlit as st
import pinecone
from datetime import datetime, date, timedelta
import openai
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Pinecone
try:
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone.Index(os.getenv("PINECONE_INDEX_NAME"))
except Exception as e:
    st.error("Error connecting to Pinecone. Please check your API key and index name.")

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embedding_fallback(text):
    return np.random.rand(1536).tolist()

def create_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"OpenAI API error in create_embedding: {str(e)}")
        return create_embedding_fallback(text)

def store_in_pinecone(data, vector, namespace):
    try:
        index.upsert(vectors=[(str(data['timestamp']), vector, data)], namespace=namespace)
    except Exception as e:
        st.error("Error storing data. Please try again later.")

def query_gpt(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return "Unable to generate analysis at this time."

def get_attendance_for_date(date, namespace="attendance"):
    try:
        query_vector = create_embedding(f"Attendance on {date}")
        results = index.query(vector=query_vector, top_k=10, namespace=namespace, include_metadata=True)
        return [r['metadata'] for r in results['matches'] if r['metadata']['entry_date'] == date.isoformat()]
    except Exception as e:
        st.error(f"Error retrieving attendance data: {str(e)}")
        return []

def calculate_working_hours(entry_time, exit_time):
    entry = datetime.strptime(entry_time, "%H:%M:%S")
    exit = datetime.strptime(exit_time, "%H:%M:%S")
    duration = exit - entry
    return duration.total_seconds() / 3600  # Convert to hours

def main():
    st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")

    st.title("🗓️ Leave Buddy: Attendance and Leave Monitoring System")

    menu = ["📅 Daily Attendance", "🏖️ Leave Request", "📊 View Attendance"]
    choice = st.sidebar.radio("Select Option", menu)

    if choice == "📅 Daily Attendance":
        st.header("📅 Daily Attendance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("👤 Employee Name")
            email = st.text_input("📧 Email ID")
        
        with col2:
            entry_date = st.date_input("📆 Date", date.today())
            entry_time = st.time_input("🕒 Entry Time")
        
        exit_time = st.time_input("🕒 Exit Time", value=None)
        
        if st.button("📝 Submit Attendance", use_container_width=True):
            if name and email and exit_time:
                timestamp = datetime.now().isoformat()
                data = {
                    "timestamp": timestamp,
                    "name": name,
                    "email": email,
                    "entry_date": entry_date.isoformat(),
                    "entry_time": entry_time.isoformat(),
                    "exit_time": exit_time.isoformat()
                }
                
                text_to_embed = f"Attendance: {name} {email} {entry_date} {entry_time} {exit_time}"
                vector = create_embedding(text_to_embed)
                store_in_pinecone(data, vector, namespace="attendance")
                
                working_hours = calculate_working_hours(entry_time.isoformat(), exit_time.isoformat())
                
                prompt = f"Analyze the attendance: Employee {name} entered on {entry_date} at {entry_time} and left at {exit_time}, working for {working_hours:.2f} hours"
                analysis = query_gpt(prompt)
                
                st.success("✅ Attendance recorded successfully!")
                st.info(f"🕒 Total hours worked: {working_hours:.2f}")
                st.info("🤖 GPT-4o-mini Analysis: " + analysis)
            else:
                st.error("❌ Please fill in all required fields.")

    elif choice == "🏖️ Leave Request":
        st.header("🏖️ Leave Request")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("👤 Employee Name")
            email = st.text_input("📧 Email ID")
            leave_from = st.date_input("📅 Leave From")
        
        with col2:
            leave_to = st.date_input("📅 Leave To")
            leave_type = st.selectbox("🏷️ Leave Type", ["Annual Leave", "Sick Leave", "Personal Leave", "Other"])
            permitted_by = st.text_input("👨‍💼 Permitted By")
        
        purpose = st.text_area("📝 Purpose of Leave")
        
        if st.button("📨 Submit Leave Request", use_container_width=True):
            if name and email and purpose:
                timestamp = datetime.now().isoformat()
                data = {
                    "timestamp": timestamp,
                    "name": name,
                    "email": email,
                    "leave_from": leave_from.isoformat(),
                    "leave_to": leave_to.isoformat(),
                    "leave_type": leave_type,
                    "purpose": purpose,
                    "permitted_by": permitted_by
                }
                
                text_to_embed = f"Leave: {name} {email} {leave_from} {leave_to} {leave_type} {purpose}"
                vector = create_embedding(text_to_embed)
                store_in_pinecone(data, vector, namespace="leave")
                
                prompt = f"Analyze the leave request: Employee {name} requested {leave_type} from {leave_from} to {leave_to} for the purpose: {purpose}"
                analysis = query_gpt(prompt)
                
                st.success("✅ Leave request submitted successfully!")
                st.info("🤖 GPT-4o-mini Analysis: " + analysis)
            else:
                st.error("❌ Please fill in all required fields.")

    elif choice == "📊 View Attendance":
        st.header("📊 View Attendance")
        
        view_date = st.date_input("📆 Select Date", date.today())
        attendance_data = get_attendance_for_date(view_date)
        
        if attendance_data:
            st.subheader(f"Attendance for {view_date}")
            for entry in attendance_data:
                working_hours = calculate_working_hours(entry['entry_time'], entry['exit_time'])
                st.write(f"👤 {entry['name']} ({entry['email']})")
                st.write(f"🕒 Entry: {entry['entry_time']}, Exit: {entry['exit_time']}")
                st.write(f"⏱️ Total hours worked: {working_hours:.2f}")
                st.write("---")
        else:
            st.info("No attendance data found for the selected date.")

if __name__ == "__main__":
    main()
