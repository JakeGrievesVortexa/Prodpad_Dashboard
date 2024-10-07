import streamlit as st
# import pandas as pd
# import requests
# import pandas as pd
# import os
# import ast
# from dotenv import load_dotenv
# from vortexasdk import Products
# import matplotlib.pyplot as plt
# from datetime import datetime,timedelta
# import numpy as np
# import seaborn as sns


st.set_page_config(
    page_title="API Usage",
    page_icon="📈",
    layout="wide",
)
#global df


ProdPad = st.Page("production_prodpad.py",title = "ProdPad Overview")

# Function to handle page navigation
def navigate_to(page_name):
    page = page_name
    main()

# Main function to control the navigation
def main():
    #st.markdown("<h1 style='text-align: left;'>Product User Dashboard: Version 1</h1>", unsafe_allow_html=True)

    pg = st.navigation([ProdPad])
    pg.run()


if __name__ == "__main__":
    main()


