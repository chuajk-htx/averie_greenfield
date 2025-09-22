import streamlit as st
#from Clearance_v3_live import live_analysis

#import Clearance_v3
from test_mock_page1 import offline_analysis
from test_mock_page2 import live_analysis

def main():
    # Custom CSS to center tabs
    st.markdown("""
        <style>
            /* Centre Page Title*/
            #contact-lens-detection {
                text-align: center;
            }
            
            /* Center the tab container */
            .stTabs [data-baseweb="tab-list"] {
                justify-content: center;
            }
            
            /* Optional: Style the tabs */
            .stTabs [data-baseweb="tab"] {
                margin: 0 10px;
                padding: 10px 20px;
                border-radius: 10px 10px 0 0;
                background-color: #f0f2f6;
                color: #333;
                border: none;
            }
            
            /* Active tab styling */
            .stTabs [aria-selected="true"] {
                background-color: #1f77b4;
                color: white;
            }
        </style>
    """, 
    unsafe_allow_html=True)
    
    st.set_page_config(page_title="Greenfield Demo App", layout="wide")
    
    st.title("Contact Lens Detection")
    
    tab1, tab2 = st.tabs(["Offline Analysis", "Live Analysis"])
    
    with tab1:
        offline_analysis()
        
    with tab2:
        live_analysis()
    
if __name__=="__main__":
    main()

