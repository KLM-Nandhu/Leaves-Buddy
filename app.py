import streamlit as st
import pinecone
from datetime import datetime, date, timedelta
from openai import OpenAI
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
    st.error(f"Error connecting to Pinecone: {str(e)}")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding_fallback(text):
    return np.random.rand(1536).tolist()

def create_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI API error in create_embedding: {str(e)}")
        return create_embedding_fallback(text)

def store_in_pinecone(data, vector, namespace):
    try:
        index.upsert(vectors=[(str(data['timestamp']), vector, data)], namespace=namespace)
    except Exception as e:
        st.error(f"Error storing data: {str(e)}")

def query_gpt(prompt):
    try:
        response = client.completions.create(
            model="gpt-4o-mini",
            prompt=prompt,
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Unable to generate analysis: {str(e)}"

def parse_date(date_string):
    if not isinstance(date_string, str):
        return None
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        return None

def parse_time(time_string):
    if not isinstance(time_string, str):
        return None
    try:
        return datetime.strptime(time_string, "%H:%M:%S").time()
    except ValueError:
        return None

def get_attendance_for_date(selected_date, namespace="attendance"):
    try:
        query_vector = create_embedding(f"Attendance on {selected_date}")
        results = index.query(vector=query_vector, top_k=10, namespace=namespace, include_metadata=True)
        filtered_results = []
        for r in results['matches']:
            entry_date = parse_date(r['metadata'].get('entry_date'))
            if entry_date == selected_date:
                filtered_results.append(r['metadata'])
        return filtered_results
    except Exception as e:
        st.error(f"Error retrieving attendance data: {str(e)}")
        return []

def calculate_working_hours(entry_time, exit_time):
    entry = parse_time(entry_time)
    exit = parse_time(exit_time)
    if entry is None or exit is None:
        return 0
    duration = datetime.combine(date.today(), exit) - datetime.combine(date.today(), entry)
    return max(0, duration.total_seconds() / 3600)  # Convert to hours, ensure non-negative

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
        
        exit_time = st.time_input("🕒 Exit Time")
        
        if st.button("📝 Submit Attendance", use_container_width=True):
            if name and email and entry_time and exit_time:
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
                entry_time = entry.get('entry_time')
                exit_time = entry.get('exit_time')
                working_hours = calculate_working_hours(entry_time, exit_time)
                st.write(f"👤 {entry.get('name', 'N/A')} ({entry.get('email', 'N/A')})")
                st.write(f"🕒 Entry: {entry_time or 'N/A'}, Exit: {exit_time or 'N/A'}")
                st.write(f"⏱️ Total hours worked: {working_hours:.2f}")
                st.write("---")
        else:
            st.info("No attendance data found for the selected date.")

if __name__ == "__main__":
    main()
