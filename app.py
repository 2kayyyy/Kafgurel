import streamlit as st

def main():
    st.title("Hello Streamlit!")
    name = st.text_input("Enter your name:")
    if st.button("Greet"):
        st.write(f"Hello mf, {name}!")

if __name__ == "__main__":
    main()