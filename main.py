import streamlit as st
from streamlit_option_menu import option_menu
import home, data_frame, pred, viz

st.set_page_config(
    page_title="AVP",
)

class Multiapp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function,
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title="AVP",
                options=['Home', 'Data Frame', 'Vizualization', 'Prediction'],
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": 'black'},
                    "nav-link": {"color": 'white', 'font-size': '20px', 'text-align': 'left', 
                                 'margin': '0px', 'nav-link-selected': {'background-color': '#02ab21'}}
                }
            )

        if app == 'Home':
            home.app()
        elif app == 'Data Frame':
            data_frame.app()
        elif app == 'Vizualization':
            viz.app()
        elif app == 'Prediction':
            pred.app()

multiapp = Multiapp()
multiapp.run()
