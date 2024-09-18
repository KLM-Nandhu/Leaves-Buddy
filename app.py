import streamlit as st
import pinecone
from datetime import datetime, date
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import io

# Set page config at the very beginning
st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")

# Load environment variables
load_dotenv()

# Constants
PINECONE_INDEX_NAME = "leave-buddy-index"

# Name-Email mappings
NAME_EMAIL_MAPPING = {
    "Nandhakumar": "nandhakumar@klmsolutions.in",
    "Dhanush": "dhanush@klmsolutions.in",
    "Shubaritha": "shubaritha@klmsolutions.in",
    "Subashree": "subashree@klmsolutions.in",
    "Prateeka": "prateeka@klmsolutions.in",
    "Akshara Shri": "akshara@klmsolutions.in"
}

# Initialize Pinecone
pinecone_initialized = False
index = None

def init_pinecone(api_key):
    global pinecone_initialized, index
    try:
        pinecone.init(api_key=api_key)
        index = pinecone.Index(PINECONE_INDEX_NAME)
        # Test the connection
        index.describe_index_stats()
        pinecone_initialized = True
        st.success(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}' successfully")
        return True
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        return False

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

def store_in_pinecone(data, vector):
    if not pinecone_initialized:
        st.warning("Pinecone is not initialized. Data will not be stored.")
        return False
    try:
        string_data = {k: str(v) if v is not None else "" for k, v in data.items()}
        index.upsert(vectors=[(string_data['timestamp'], vector, string_data)])
        return True
    except Exception as e:
        st.error(f"Error storing data in Pinecone: {str(e)}")
        return False

def query_gpt(prompt):
    try:
        response = client.chat_completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing attendance and leave data."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"Unable to generate analysis: {str(e)}"

def calculate_working_hours(entry_time, exit_time):
    try:
        entry = datetime.strptime(entry_time, "%H:%M:%S")
        exit = datetime.strptime(exit_time, "%H:%M:%S")
        duration = exit - entry
        return max(0, duration.total_seconds() / 3600)
    except ValueError:
        return 0

def fetch_attendance(employee_name, from_date, to_date):
    try:
        query_vector = create_embedding(f"Attendance of {employee_name} from {from_date} to {to_date}")
        if query_vector:
            results = index.query(
                vector=query_vector,
                top_k=10,
                include_metadata=True
            )
            attendance_data = [
                r['metadata'] for r in results['matches']
                if r['metadata'].get('type') == 'attendance'
                and from_date <= r['metadata'].get('entry_date', '') <= to_date
                and r['metadata'].get('name') == employee_name
            ]
            return attendance_data
        return []
    except Exception as e:
        st.error(f"An error occurred while querying Pinecone: {str(e)}")
        return []

def download_to_excel(data, employee_name):
    df = pd.DataFrame(data)
    filename = f"{employee_name}_attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    
    # Convert dataframe to excel in-memory buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    
    st.download_button(
        label="📥 Download Excel File",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def main():
    st.title("🗓️ Leave Buddy: Attendance and Leave Monitoring System")

    # Initialize Pinecone connection
    if not pinecone_initialized:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key:
            init_pinecone(pinecone_api_key)
        else:
            st.error("Pinecone API key not found in environment variables.")
            return

    menu = ["📅 Daily Attendance", "🏖️ Leave Request", "📊 View Attendance"]
    choice = st.sidebar.radio("Select Option", menu)

    if choice == "📅 Daily Attendance":
        st.header("📅 Daily Attendance")

        col1, col2 = st.columns(2)
        
        with col1:
            name = st.selectbox("👤 Employee Name", list(NAME_EMAIL_MAPPING.keys()))
            email = NAME_EMAIL_MAPPING[name]
            st.text_input("📧 Email ID", value=email, disabled=True)
        
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
                        "type": "attendance",
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
                        if store_in_pinecone(data, vector):
                            working_hours = calculate_working_hours(entry_time.isoformat(), exit_time.isoformat())
                            
                            # GPT-4 Analysis
                            prompt = f"Analyze the attendance: Employee {name} entered on {entry_date} at {entry_time} and left at {exit_time}, working for {working_hours:.2f} hours"
                            st.info("Generating analysis with GPT-4...")
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
            name = st.selectbox("👤 Employee Name", list(NAME_EMAIL_MAPPING.keys()))
            email = NAME_EMAIL_MAPPING[name]
            st.text_input("📧 Email ID", value=email, disabled=True)
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
                        "type": "leave",
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
                        if store_in_pinecone(data, vector):
                            
                            # GPT-4 Analysis
                            prompt = f"Analyze the leave request: Employee {name} requested {leave_type} from {leave_from} to {leave_to} for the purpose: {purpose}"
                            st.info("Generating analysis with GPT-4...")
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

        # Date Range Selection
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input("📆 From", date.today())
        with col2:
            to_date = st.date_input("📆 To", date.today())

        # Employee Selection
        employee_name = st.selectbox("👤 Select Employee", list(NAME_EMAIL_MAPPING.keys()))

        # View and Download Buttons
        view_button, download_button = st.columns(2)

        if view_button.button("👁️ View Attendance"):
            # Fetch and display attendance data
            attendance_data = fetch_attendance(employee_name, from_date.isoformat(), to_date.isoformat())
            if attendance_data:
                st.subheader(f"Attendance for {employee_name} from {from_date} to {to_date}")
                for entry in attendance_data:
                    entry_time = entry.get('entry_time', 'N/A')
                    exit_time = entry.get('exit_time', 'N/A')
                    working_hours = calculate_working_hours(entry_time, exit_time)
                    st.write(f"👤 {entry.get('name', 'N/A')} ({entry.get('email', 'N/A')})")
                    st.write(f"🕒 Entry: {entry_time}, Exit: {exit_time}")
                    st.write(f"⏱️ Total hours worked: {working_hours:.2f}")
                    st.write("---")
            else:
                st.info(f"No attendance data found for {employee_name} in the selected date range.")

        if download_button.button("📥 Download Attendance"):
            # Fetch data and download as Excel
            attendance_data = fetch_attendance(employee_name, from_date.isoformat(), to_date.isoformat())
            if attendance_data:
                download_to_excel(attendance_data, employee_name)
            else:
                st.warning(f"No attendance data available for download for {employee_name} from {from_date} to {to_date}.")

if __name__ == "__main__":
    main()
