import streamlit as st
import pinecone
from datetime import datetime, date
import openai

# Initialize Pinecone
pinecone.init(api_key="YOUR_PINECONE_API_KEY")
index = pinecone.Index("leave-buddy-index")

# Initialize OpenAI
openai.api_key = "YOUR_OPENAI_API_KEY"

# Function to create embeddings using OpenAI
def create_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Function to store data in Pinecone
def store_in_pinecone(data, vector, namespace):
    index.upsert(vectors=[(str(data['timestamp']), vector, data)], namespace=namespace)

# Function to query GPT-4o-mini
def query_gpt(prompt):
    response = openai.Completion.create(
        model="gpt-4o-mini",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Streamlit app
def main():
    st.set_page_config(page_title="Leave Buddy", page_icon="🗓️", layout="wide")

    st.title("🗓️ Leave Buddy: Attendance and Leave Monitoring System")

    menu = ["📅 Daily Attendance", "🏖️ Leave Request"]
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
        
        if st.button("📝 Submit Attendance", use_container_width=True):
            if name and email:
                timestamp = datetime.now().isoformat()
                data = {
                    "timestamp": timestamp,
                    "name": name,
                    "email": email,
                    "entry_date": entry_date.isoformat(),
                    "entry_time": entry_time.isoformat()
                }
                
                # Create embedding and store in Pinecone
                text_to_embed = f"Attendance: {name} {email} {entry_date} {entry_time}"
                vector = create_embedding(text_to_embed)
                store_in_pinecone(data, vector, namespace="attendance")
                
                # Query GPT-4o-mini
                prompt = f"Analyze the attendance: Employee {name} entered on {entry_date} at {entry_time}"
                analysis = query_gpt(prompt)
                
                st.success("✅ Attendance recorded successfully!")
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
                
                # Create embedding and store in Pinecone
                text_to_embed = f"Leave: {name} {email} {leave_from} {leave_to} {leave_type} {purpose}"
                vector = create_embedding(text_to_embed)
                store_in_pinecone(data, vector, namespace="leave")
                
                # Query GPT-4o-mini
                prompt = f"Analyze the leave request: Employee {name} requested {leave_type} from {leave_from} to {leave_to} for the purpose: {purpose}"
                analysis = query_gpt(prompt)
                
                st.success("✅ Leave request submitted successfully!")
                st.info("🤖 GPT-4o-mini Analysis: " + analysis)
            else:
                st.error("❌ Please fill in all required fields.")

if __name__ == "__main__":
    main()
