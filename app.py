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

def create_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def store_in_pinecone(data, vector, namespace):
    try:
        # Convert all values to strings to avoid type issues
        string_data = {k: str(v) if v is not None else "" for k, v in data.items()}
        index.upsert(vectors=[(string_data['timestamp'], vector, string_data)], namespace=namespace)
        return True
    except Exception as e:
        st.error(f"Error storing data in Pinecone: {str(e)}")
        return False

def query_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing attendance and leave data."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate analysis: {str(e)}"

def calculate_working_hours(entry_time, exit_time):
    try:
        entry = datetime.strptime(entry_time, "%H:%M:%S")
        exit = datetime.strptime(exit_time, "%H:%M:%S")
        duration = exit - entry
        return max(0, duration.total_seconds() / 3600)  # Convert to hours, ensure non-negative
    except ValueError:
        return 0

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
                with st.spinner("Processing attendance data..."):
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
                    st.info("Creating embedding...")
                    vector = create_embedding(text_to_embed)
                    
                    if vector:
                        st.info("Storing data in Pinecone...")
                        if store_in_pinecone(data, vector, namespace="attendance"):
                            working_hours = calculate_working_hours(entry_time.isoformat(), exit_time.isoformat())
                            
                            prompt = f"Analyze the attendance: Employee {name} entered on {entry_date} at {entry_time} and left at {exit_time}, working for {working_hours:.2f} hours"
                            st.info("Generating analysis...")
                            analysis = query_gpt(prompt)
                            
                            st.success("✅ Attendance recorded successfully!")
                            st.info(f"🕒 Total hours worked: {working_hours:.2f}")
                            st.info("🤖 GPT-4 Analysis: " + analysis)
                        else:
                            st.error("Failed to store data in Pinecone.")
                    else:
                        st.error("Failed to create embedding.")
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
                with st.spinner("Processing leave request..."):
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
                    st.info("Creating embedding...")
                    vector = create_embedding(text_to_embed)
                    
                    if vector:
                        st.info("Storing data in Pinecone...")
                        if store_in_pinecone(data, vector, namespace="leave"):
                            prompt = f"Analyze the leave request: Employee {name} requested {leave_type} from {leave_from} to {leave_to} for the purpose: {purpose}"
                            st.info("Generating analysis...")
                            analysis = query_gpt(prompt)
                            
                            st.success("✅ Leave request submitted successfully!")
                            st.info("🤖 GPT-4 Analysis: " + analysis)
                        else:
                            st.error("Failed to store data in Pinecone.")
                    else:
                        st.error("Failed to create embedding.")
            else:
                st.error("❌ Please fill in all required fields.")

   elif choice == "📊 View Attendance":
        st.header("📊 View Attendance")
        
        view_date = st.date_input("📆 Select Date", date.today())
        if st.button("View Attendance"):
            with st.spinner("Fetching attendance data..."):
                query_vector = create_embedding(f"Attendance on {view_date}")
                if query_vector:
                    try:
                        if index is None:
                            st.error("Pinecone index is not properly initialized. Please check your configuration.")
                        else:
                            results = index.query(
                                vector=query_vector,
                                top_k=10,
                                namespace="attendance",
                                include_metadata=True
                            )
                            attendance_data = [r['metadata'] for r in results['matches'] if r['metadata'].get('entry_date') == view_date.isoformat()]
                            
                            if attendance_data:
                                st.subheader(f"Attendance for {view_date}")
                                for entry in attendance_data:
                                    entry_time = entry.get('entry_time', '')
                                    exit_time = entry.get('exit_time', '')
                                    working_hours = calculate_working_hours(entry_time, exit_time)
                                    st.write(f"👤 {entry.get('name', 'N/A')} ({entry.get('email', 'N/A')})")
                                    st.write(f"🕒 Entry: {entry_time or 'N/A'}, Exit: {exit_time or 'N/A'}")
                                    st.write(f"⏱️ Total hours worked: {working_hours:.2f}")
                                    st.write("---")
                            else:
                                st.info("No attendance data found for the selected date.")
                    except Exception as e:
                        st.error(f"An error occurred while querying Pinecone: {str(e)}")
                        st.error("Please check your Pinecone configuration and ensure the index is properly set up.")
                else:
                    st.error("Failed to create query embedding. Please try again.")
                    
if __name__ == "__main__":
    main()
