import streamlit as st

def main():
    offline = st.Page('Clearance_v3.py')
    live = st.Page('live.py')

    pages = {"Application Mode": [offline,live]}

    pg = st.navigation(pages, expanded=True)
    pg.run()

if __name__=="__main__":
    main()

