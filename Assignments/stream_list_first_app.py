
import streamlit as st


st.title("Adding Two Numbers")
st.title("Test Page")
num1 = st.number_input("First Number: ", value=0)
num2 = st.number_input("Second Number: ", value=0)

if st.button("Calculate Sum"):
    result = num1 + num2
    st.text("Sum is " + str(result))
