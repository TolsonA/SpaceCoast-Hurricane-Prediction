from joblib import load
import pandas as pd
import streamlit as st
import numpy as np

reg_model = load('models/no_tune_regression_model.joblib')
rf_model = load('models/rf_model.joblib')


if __name__ == "__main__":
    
  def intro():

      st.write("# SLD 45 Hurricane Prediction!")
      st.sidebar.success("Select a page above.")

      st.markdown(
        """
        Provide analysis of hurricane trends over time and 
        prediction of future hurricanes. 
        
    """
    )

      st.image('Images/Area_pic.png', caption='Area of Interest', use_column_width=True)

  def model_demo():


    
    def predict(features):
        features = np.array(features).reshape(1, -1)  # Ensure the features are in the correct shape
        prediction = reg_model.predict(features)
        return prediction[0]

    st.title('AMO Anomaly Model')
    st.write(
        """
        This model will predict the probability of seeing a hurricane in our area based on the AMO anomaly.
        \n Keep in mind even with higher anomalies it does not guarantee a hurricane, but rather increases
        the chances of getting one. 
"""
    )
    st.sidebar.title('AMO Anomaly Forecast Parameter')
    feature = st.sidebar.slider("AMO_Annual", -0.50, 0.51, 0.0)
    features = [1.0, feature]

    prediction = predict(features)
    percentage = prediction * 100
        
    # prediction_label = {0: "No", 1: "Yes"}.get(prediction[0], "Unknown")    


    # st.sidebar.title("Input Parameters")
    # param1 = st.sidebar.slider("Sahel_Annual", -4.011, 6.95, 0.0)
    # param2 = st.sidebar.slider("AMO_Annual", -0.44, 0.33, 0.1)
    # param3 = st.sidebar.slider("ENSO_Annual", -1.40, 1.77, -0.02)
    # param4 = st.sidebar.slider("TEMP_Annual", 77.2, 81.5, 79.0)
    # param5 = st.sidebar.slider("AMM_sst", -4.64, 4.64, 0.2)
    # param6 = st.sidebar.slider("Nina_index", -1.28, 1.83, -0.1)
    # param7 = st.sidebar.slider("NAO_Jones", -1.84, 1.75, 0.4)
    # param8 = st.sidebar.slider("rh_value", 28.12, 50.4, 40.0)
    # param9 = st.sidebar.slider("height_x", 1406.63, 1467.67, 1440.0)
    # param10 = st.sidebar.slider("trop_pressure", 95.45, 106.79, 100.0)
    # param11 = st.sidebar.slider("Off_Coast_Pressure", 96.82, 110.7, 102.0)
    # param12 = st.sidebar.slider("height_y", 1406.63, 1467.67, 1440.0)
    # param13 = st.sidebar.slider("geo_height_offcoast", 5795.05, 5862.43, 5832.0)

    # input_data = np.array([[param1, param2, param3, param4, param5, param6, param7,
    #                         param8, param9, param10, param11, param12, param13]])
    # scaler = StandardScaler()
    # scaled_input_data = scaler.fit_transform(input_data)

    # prediction = loaded_model.predict(input_data)
    # prediction_label = {0: "No", 1: "Yes"}.get(prediction[0], "Unknown")

    st.markdown(f"<h1 style='font-size: 24px;'>Probability of getting a hurricane: {percentage:.2f}%</h1>", unsafe_allow_html=True)
    
    st.title('Hurricane Likelihood based Sea Level Pressure Model')

    def predict2(features2):
        #features2 = np.array(features2).reshape(1, -1)  # Ensure the features are in the correct shape
        prediction2 = rf_model.predict(features2)
        return prediction2
    
    st.sidebar.title('Sea Level Pressure Forecast Parameters')
    feature3 = st.sidebar.slider("Nassau_slp", -1.46, 2.06, -0.1)
    feature4 = st.sidebar.slider("Charleston_slp", -2.48, 1.97, -0.5)
    feature5 = st.sidebar.slider("Meridia_slp", -3.43, 1.83, -0.3)

    features2 = [[feature3, feature4, feature5]]

    prediction2 = predict2(features2)
    prediction_label = {0: "No", 1: "Yes"}.get(prediction2[0], "Unknown")
    st.markdown(f"<h1 style='font-size: 24px;'>Will a hurricane move towards us? Probably {prediction_label}</h1>", unsafe_allow_html=True)
    st.write(
        """
        This model will predict the likelihood of a hurricane moving into our area based on sea level pressures.
        \n This model is not perfect! It is a pretty good indicator of high enough pressure to steer hurricanes away from this area. 
"""
    )
    st.write("Adjust the parameters in the sidebar to see the model prediction update in real-time.")


def hurricane_analysis():
    import streamlit as st
    import time
    import numpy as np 
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}') 

    st.write('We will look at Atlantic Multidecadal Oscillation (AMO) temperature anomaly influence on hurricanes for our area.')
    st.write('Below is a visual of the AMO region')
    st.image('Images/AMO_image.png', caption='Atlantic Multidecadal Oscillation (AMO) Region', use_column_width=True)
    st.write('Below is an image showing the AMO anomalies with the trend of hurricanes since 1850.')
    st.image('Images/1850to2022_hurricanes.png', caption='Area of Interest', use_column_width=True)  
    st.write('Below is an image showing the AMO anomalies with the trend of hurricanes since 1950.')      
    st.image('Images/Hurricane_trends.png', caption='Area of Interest', use_column_width=True)


    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.

def data_frame_demo():

    st.markdown(f"# {list(page_names_to_funcs.keys())[3]}")
    st.write(
        """
        This data is provided to help leaders make decisions. 
"""
    )
    st.markdown('Hurricanes moving up the coast (from S-SE)\nMost Dangerous')
    st.markdown('Hurricanes coming from SSW-West\nLess Dangerous')
    st.write('Historical tracks during August')
    st.image('Images/off_coast_aug.png', caption='August Cyclone Tracks', use_column_width=True)
    st.write('Historical tracks during September')
    st.image('Images/off_coast_sep.png', caption='September Cyclone Tracks', use_column_width=True)
    st.write('Historical tracks during October')
    st.image('Images/off_coast_oct.png', caption='October Cyclone Tracks', use_column_width=True)

page_names_to_funcs = {
    "Overview": intro,
    "Hurricane Analysis": hurricane_analysis,
    "Model Predictions": model_demo,
    "Decision Matrix": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
  # st.set_option('deprecation.showPyplotGlobalUse', False)
  # DATA_PATH = "models/data_reader.joblib"
        
  # st.title('Cyclone Trendline') 
    
    
  # loaded_no_filter_trend.plot_cyclone_data()
  # st.pyplot()
    
  # df_data = loaded_data_reader.all_data
  # st.dataframe(df_data)

  # df_numcyclones = load("models/df_numcyclones.joblib")

  # dict_numcyclones = df_numcyclones.set_index('Year').to_dict()['Cyclone']
  # list_from_df = list(df_numcyclones.itertuples(index=False, name=None))

  # # Get the value corresponding to the selected year
  # #selected_value = dict_numcyclones.get(year, 'Value not found')

  # # Display the selected value
  # # st.write(f'Selected year: {year}')
  # # st.write(f'Corresponding value: {selected_value}')

  # # Use widgets' returned values in variables
  # for i in range(int(st.number_input('Num:'))): foo()
  # if st.sidebar.selectbox('I:',['f']) == 'f': b()
  # my_slider_val = st.slider('Quinn Mallory', 1, 88)
  # st.write(slider_val)

  # # Insert containers separated into tabs:
  # tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
  # tab1.write("this is tab 1")
  # tab2.write("this is tab 2")

  # # You can also use "with" notation:
  # with tab1:
  #   st.radio('Select one:', [1, 2])

  # df_numcyclones['Year'] = df_numcyclones['Year'].astype('int')
  # num_cyclones = df_numcyclones['Cyclone'].to_list()

  # st.dataframe(df_numcyclones, width=180, hide_index=True)

  # st.select_slider(label='Slide me :sunglasses:'
  #             , min_value=df_numcyclones['Year'].min()
  #             , max_value=df_numcyclones['Year'].max()
  #             # , value=num_cyclones[0]
  #             , help="i need help too")

  # st.image('Images/off_coast_sep.png', caption="September Tracks")