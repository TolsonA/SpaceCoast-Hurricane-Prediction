from joblib import load
import pandas as pd
import streamlit as st
import numpy as np

reg_model = load('models/initial_regression_model.joblib')
rf_model = load('models/rf_model.joblib')

st.markdown(
    """
    <style>
    .big-font {
        font-size:50px !important;
        color: #FFFFFF;
    }
    .medium-font {
        font-size:30px !important;
        color: #4BFF4B;
    }
    .small-font {
        font-size:20px !important;
        color: #4BFFFF;
    }
    .medium-font2 {
        font-size:30px !important;
        color: #FF0000}
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    
  def intro():

    st.write("# SLD 45 Hurricane Prediction!")
    st.sidebar.success("Select a page above.")

    st.markdown('<p class="medium-font">Provide analysis of hurricane trends over time and \
                  prediction of future hurricanes.</p?', unsafe_allow_html=True)
    st.markdown('Presentation by Andrew Tolson')
    st.image('Images/Area_pic.png', caption='Area of Interest', use_column_width=True)

  def model_demo():


    
    def predict(features):
        features = np.array(features).reshape(1, -1)  # Ensure the features are in the correct shape
        prediction = reg_model.predict(features)
        return prediction[0]

    st.markdown('<p class="big-font">AMO Anomaly Model</p?', unsafe_allow_html=True)
    st.write(
        """
        This model will predict the probability of seeing a hurricane in our area based on the AMO anomaly.
        \n Keep in mind even with higher anomalies it does not guarantee a hurricane, but rather increases
        the chances of getting one. 
"""
    )
    st.sidebar.title('AMO Anomaly Forecast Parameter')
    feature = st.sidebar.slider("AMO_Annual", -0.50, 0.51, 0.3)
    features = [1.0, feature]

    prediction = predict(features)
    percentage = prediction * 100
        
    st.markdown(f'<p class="medium-font">Probability of getting a hurricane: {percentage:.2f}%</p?', unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Hurricane Likelihood based on Sea Level Pressure/AMO Model</p?', unsafe_allow_html=True)

    def predict2(features2):
        prediction2 = rf_model.predict(features2)
        return prediction2
    
    st.sidebar.title('Sea Level Pressure Forecast Parameters')
    feature3 = st.sidebar.slider("Nassau_slp", -1.46, 2.06, -0.14)
    feature4 = st.sidebar.slider("Charleston_slp", -2.48, 1.97, -0.6)
    feature5 = st.sidebar.slider("Meridia_slp", -3.43, 1.83, -0.3)
    feature6 = st.sidebar.slider("AMO_Annual", -0.50, 0.51, 0.34)

    features2 = [[feature3, feature4, feature5, feature6]]

    prediction2 = predict2(features2)
    prediction_label = {0: "No", 1: "Yes"}.get(prediction2[0], "Unknown")
    st.markdown(f'<p class="medium-font">Will a hurricane move towards us? Probably {prediction_label}</p?', unsafe_allow_html=True)
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
    st.markdown('<p class="big-font">Hurricane Analysis</p?', unsafe_allow_html=True)

    st.write('<P class="small-font">We will look at Atlantic Multidecadal Oscillation (AMO)\
              temperature anomaly influence on hurricanes for our area.</p?>', unsafe_allow_html=True)
    st.markdown('Null hypothesis: AMO has no effect on hurricanes impacting our area. \
                \n Alternate hypothesis: AMO anomalies are statistically significant in determining hurricane impacts\
                to our area.')
    st.write('Below is a visual of the AMO region')
    st.image('Images/AMO_image.png', caption='Atlantic Multidecadal Oscillation (AMO) Region', use_column_width=True)
    st.markdown('Below is an image showing the AMO anomalies with the trend of hurricanes since 1850. \
                \n Notice the correlation between hurricanes(red line) and AMO anomalies(blue line)')
    st.image('Images/1850to2022_hurricanes.png', caption='Area of Interest', use_column_width=True)  
    st.markdown('Below is an image showing the AMO anomalies with the trend of hurricanes since 1950.\
                \n We see a significant relation between AMO anomalies and hurricanes impacting this area.')      
    st.image('Images/Hurricane_trends.png', caption='Area of Interest', use_column_width=True)
    st.markdown('Below is an image showing the results of a logistic regression analysis. We can see that \
                there is a significant change for positive AMO anomalies. P-value being .01! \
                \n The top of the chart shows hurricanes that have happened and you can see how many \
                are to the right of the 0/avg line.')      
    st.image('Images/logistic_reg_chart.png', caption='LR Chart', use_column_width=True)


def data_frame_demo():

    st.markdown('<p class="big-font">Decision Recommendations</p?', unsafe_allow_html=True)
    st.write(
        """
        This data is provided to help leaders make decisions. 
"""
    )
    st.markdown('<p class="medium-font2">Hurricanes moving up the coast (from S-SE) = Most Dangerous</p?', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Hurricanes coming from SSW-West = Less Dangerous</p?', unsafe_allow_html=True)
    st.write('Breakdown on cyclones per month for our area')
    st.image('Images/monthly_avg.png', caption='Monthly Breakdown', use_column_width=True)
    st.write('Historical tracks during August')
    st.image('Images/off_coast_aug.png', caption='August Cyclone Tracks', use_column_width=True)
    st.write('Historical tracks during September')
    st.image('Images/off_coast_sep.png', caption='September Cyclone Tracks', use_column_width=True)
    st.write('Historical tracks during October')
    st.image('Images/off_coast_oct.png', caption='October Cyclone Tracks', use_column_width=True)

def last_slide():
    st.markdown('<p class="big-font">Information</p?', unsafe_allow_html=True)

    st.markdown('Project completed by Andrew Tolson')
    st.markdown('My github repository is at https://github.com/TolsonA/SpaceCoast-Hurricane-Prediction')
    st.markdown('The image below is interesting if you find yourself wanting to learn more. \
                \nMonsoon season in west Africa is very much related to hurricanes impacting \
                the United States. This is closely related to the AMO anomalies.')
    st.image('Images/W Sahel Wet vs Dry years.gif', caption='September Cyclone Tracks', use_column_width=True)

page_names_to_funcs = {
    "Overview": intro,
    "Hurricane Analysis": hurricane_analysis,
    "Model Predictions": model_demo,
    "Decision Matrix": data_frame_demo,
    "Information": last_slide
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()