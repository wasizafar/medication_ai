import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai

# Config API Key
print(os.getcwd())

api = st.secrets["auth_Key"]
genai.configure(api_key = api)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# load the trained models
p_model = joblib.load('models/xgboost/com_xgboost.pkl')
le_model = joblib.load('models/xgboost/LabelEncoder.pkl')

# list of symptoms
symptom_list = []

with open('common_diseases.txt') as file:
    symptom_list = [line.rstrip() for line in file]

# streamlit app
st.set_page_config(page_title='MEDICATION AI', page_icon='🩺')

# Sidebar for nevigation
page = st.sidebar.radio('Navigation',['Disease prediction', 'AI doctor'])
age = [i for i in range(1,65)]
personal_details = {
    'name' : None,
    'age' : None,
    'gender' : None,
    'disease': None
}

if page == 'Disease prediction':
    predicted_disease = ''
    st.title('Medication AI')
    with st.form(key='disease'):
        # enter the personal details
        personal_details['name'] = st.text_input('enter your name: ')
        personal_details['age'] = st.selectbox('Age: ', age)
        personal_details['gender'] = st.selectbox('Gender', ['Male','Female'])


        st.write('Select the symptoms you are experiencing:')

        # Dictionary with all symptoms set to 0
        user_input = {symptom: 0 for symptom in symptom_list}

        # multi-select for symptoms
        selectd_symptoms = st.multiselect('Select symptoms:', symptom_list)

        # Update selected synotoms to 1
        for symtom in selectd_symptoms:
            symtom = symtom.strip()
            if symtom in user_input:
                user_input[symtom] = 1
            else:
                st.subheader('warning some symptoms are not recognized symptons')
        
        
        symptoms = st.form_submit_button()

        if symptoms:
            if not selectd_symptoms:
                st.write('Please fill the symptoms Box')
            else:
                input_data = pd.DataFrame([user_input])
                predicted_disease = p_model.predict(input_data)
                predicted_disease = le_model.inverse_transform(predicted_disease)

                personal_details['disease'] = predicted_disease

                st.metric('based on you given symptoms the disease is :', predicted_disease[0])

                # Function to get medical recommendations
                def get_medical_advice(disease, name, age, gender):
                    prompt = f"""
                    A patient named {name}, aged {age}, gender {gender}, has been diagnosed with {disease}.
                    Provide the following:
                    1. Explain about disease
                    2. Recommended medicine and does based on age (if not critical)
                    3. Doctor consultation recommendation (if needed)
                    4. Foods to eat & avoid
                    5. Key precautions
                    Keep it short and clear
                    """
                    response = model.generate_content(prompt)
                    return response.text
                
                # example usage
                disease = predicted_disease
                name = personal_details['name']
                age = personal_details['age']
                gender = personal_details['gender']
                disease = personal_details['disease']

                advice = get_medical_advice(disease=disease, name=name, age=age, gender=gender)

                st.caption(advice)

if page == 'AI doctor':
    st.title("🩺 AI Medical Chatbot")

    st.write("💬 Chat with an AI doctor about symptoms, diseases, or health advice.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your AI doctor. How can I help you today?"}
        ]

    # Display previous chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask me anything about your health...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Check if input is a medical question or just a greeting
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you", "what's up"]

        if user_input.lower().strip() in greetings:
            response = (
                "**Hello!** 👨‍⚕️ I'm your AI doctor.\n"
                "I'm here to assist you with any medical concerns. Please describe your symptoms, ask about a condition, or seek advice on health and wellness."
            )
        else:
            # AI Medical Response
            prompt = f"""
            You are a medical AI chatbot. Answer the following health-related question concisely:
            
            **User Query:** {user_input}
            
            Provide a response that includes:
            - Possible condition or explanation
            - Recommended medicine and does based on age (if applicable)
            - Foods to eat & avoid
            - Key precautions
            - Whether a doctor visit is necessary.
            
            Keep it short and clear.
            """
            
            response = model.generate_content(prompt).text.strip()

        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(response)



