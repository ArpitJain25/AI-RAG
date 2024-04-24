import os
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="RAG Application by APJ",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# sidebar for navigation
with st.sidebar:
      selected = option_menu('Menu',
                           ['About Me',
                            'BJP Manifesto'],
                           menu_icon='rocket',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Election 2024 - Bhartiya Janta Party (BJP) Manifesto Search
if selected == 'BJP Manifesto':

    # page title
    st.title('Election 2024 - Bhartiya Janta Party (BJP) Manifesto Search')

    # getting the input data from the user
    col1 = st.columns(1)

    with col1:
        Question = st.text_input('Question')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Check'):

        user_input = [Question]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)
