import streamlit as st 
import re
from rag import save_text_file, reply, get_chat_history

# This file contains the web application part

st.set_page_config(page_title='File Uploader')

def is_valid_string(input_string):
    return re.match("^[a-zA-Z0-9]*$", input_string) is not None

def ch_retriever(user_id):
    l = get_chat_history(user_id)
    i=0
    ret = []
    for li in l:
        if i%2==0:
            start = li.find('content=') + len('content=')
            end = li.rfind(')')
            content = li[start:end]
            ret.append({"role": "user", "content": content})
        else:
            start = li.find('content=') + len('content=')
            end = li.rfind(')')
            content = li[start:end]
            ret.append({"role": "assistant", "content": content})  
        i = i+1
    return ret  

# Text input
user_id = st.sidebar.text_input(label='User ID (no spaces or special characters)')

# Validate the input string
if user_id and is_valid_string(user_id) :
    st.sidebar.write(f'Valid user id: {user_id}')
    
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if user_id not in st.session_state.messages:
        st.session_state.messages[user_id]=ch_retriever(user_id)
 
    for message in st.session_state.messages[user_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_query := st.chat_input("Ask your queries here"):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages[user_id].append({"role": "user", "content": user_query})
        with st.spinner("Processing.."):
            response = reply(user_query,user_id)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages[user_id].append({"role": "assistant", "content": response})    
       
    file_uploaded = st.sidebar.file_uploader(label='Upload your text file',type=['txt'])
    if file_uploaded and st.sidebar.button('Upload file'):
        file_name = file_uploaded.name
        if file_uploaded.name.endswith('.txt'):
            text_content = file_uploaded.read().decode('utf-8')
            with open("files/"+user_id+file_name,"w") as nf:
                nf.write(text_content)
            with st.spinner('Processing...'):
                save_text_file('files/'+user_id+file_name,user_id)
                st.sidebar.success('process finished') 
            with st.chat_message("Docs"):
                st.markdown(file_name+" saved.")
            st.session_state.messages[user_id].append({"role": "Docs", "content": file_name+" saved."})
            
        else:
            st.write("Only .txt files are allowed!") 
else:
    st.sidebar.write('Please enter a valid user ID')
