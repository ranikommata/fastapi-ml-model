import streamlit as st
import requests
mathmark=st.number_input("Enter maths mark")
scimark=st.number_input("Enter science mark")
engmark=st.number_input("Enter eng mark")
marks=[mathmark,scimark,engmark]
if st.button("Predict Result"):
    response=requests.post("http://localhost:8000/predict",json={"marks":marks})
    if response.status_code==200:
        result=response.json().get("prediction")
        if result==1:
            st.success("Passed")
        else:
            st.error("Fail")