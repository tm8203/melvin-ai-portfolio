import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from textblob import TextBlob
import os

# Initialize sidebar state
#if "sidebar_state" not in st.session_state:
    #st.session_state.sidebar_state = "collapsed"  # Default state
    
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "collapsed"  #Start collapsed only on first load

# Page configuration
#st.set_page_config(
    #page_title="Melvin Tejada's AI Portfolio",
    #layout="wide",
    #initial_sidebar_state=st.session_state.sidebar_state  # Use session state to track sidebar behavior
#)

st.set_page_config(
    page_title="Melvin Tejada's AI Portfolio",
    layout="wide",
    initial_sidebar_state=st.session_state.get("sidebar_state", "expanded")  # Dynamic state
)

# Inject custom CSS
st.markdown("""
    <style>
        /* Left justify button text */
        .stButton button {
            text-align: left !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Portfolio_Home_Files/synthetic_cloud_pricing_dataset.csv")

data = load_data()

# Sentiment Analysis Function
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Sidebar Structure with session state for navigation
st.sidebar.title("Melvin Tejada's AI Portfolio")
if "page" not in st.session_state:
    st.session_state.page = "about-me"

#def navigate(page_name):
    #if st.session_state.page != page_name:
        #st.session_state.page = page_name
        #st.query_params.update({"page": page_name})  # Update query params to force a refresh

def navigate(page_name):
    st.session_state.page = page_name
    st.session_state.sidebar_state = "collapsed"  # Collapse sidebar on selection
    st.session_state.needs_rerun = True  # Set flag instead of calling st.rerun()

#def navigate(page_name):
    #if st.session_state.page != page_name:
        #st.session_state.page = page_name
        #st.session_state.sidebar_state = "collapsed"  # Collapse sidebar after button click
        #st.session_state.needs_rerun = True  # Use flag instead of st.rerun()




st.sidebar.button("About Me", on_click=navigate, args=("about-me",))
st.sidebar.markdown("## Scenario:<br>Analyze cloud spend by customer segment using AI/ML/DL", unsafe_allow_html=True)
st.sidebar.button("First I Create Synthetic Data. Then...", on_click=navigate, args=("creating-synthetic-data",))
st.sidebar.button("[**interactive**] Use Scalar Regression to Forecast Spend", on_click=navigate, args=("scalar-regression",))
st.sidebar.button("[**interactive**] Use a Variational Autoencoder (VAE) to Detect Anomalies", on_click=navigate, args=("vae-anomaly-detection",))
st.sidebar.button("[**interactive**] Use Natural Language Processing (NLP) to Analyze Customer Insights", on_click=navigate, args=("nlp-customer-insights",))
st.sidebar.markdown("## More Fun Generative AI and Deep Learning Samples")
st.sidebar.button("Create Your Own Picture Filter with Neural Style Transfer (NST)", on_click=navigate, args=("nst-filters",))
st.sidebar.button("Detect Image Features with a Convolutional Neural Network (CNN)", on_click=navigate, args=("cnn-feature-detection",))
st.sidebar.button("[**interactive**] Analyze Sound with Spectrogram Insights", on_click=navigate, args=("sound-analysis",))

# Display relevant section
page = st.session_state.page

if page == "about-me":
    st.title("About Me")
    
    # Display your photo
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("Portfolio_Home_Files/photo.jpg", caption="Melvin Tejada", width=150)  # Adjust the width as needed
    
    # Display your blurb in the second column
    with col2:
        st.markdown("""
            Welcome to my AI portfolio! I'm **Melvin Tejada**, an experienced professional with a passion for leading in technical product management, 
            program management, and advanced analytics. I focus on delivering innovative solutions that bridge cutting-edge AI with impactful business outcomes.  
            
            Explore this portfolio to see a small sample of my work with data and AI/ML/DL.
            
            **_Pro Tip: this portfolio pairs well with a sense of curiosity_** :)
            
            My portfolio is split into two sections:  
            - The first section walks through analyzing cloud spend with a series of **interactive** AI models (~ one minute or less each)  
            - The second offers some cool and also **interactive** models that create image filters, detect images, and analyze sound visuaully (also each a minute or less)!
            Have fun!
            
            Feel free to connect via LinkedIn (link below) or email me at **tejada.melvin@gmail.com**. 
            """, unsafe_allow_html=True)
    
        st.write("[LinkedIn](https://www.linkedin.com/in/melvin-tejada/) | [Resume](https://github.com/tm8203/melvin-ai-portfolio/blob/main/melvins-resume-2025.pdf)")



elif page == "creating-synthetic-data":
    st.title("Creating Synthetic Data")
    st.write("**Description:** To simulate a real business scenario of analyzing pricing and spend on cloud services, I fully synthesized this dataset of 500 customer accounts using custom Python scripts. It generates realistic cloud data simulating usage patterns, including forecasts, service usage, and satisfaction metrics (data sample below).")
    st.dataframe(data.head(10))

elif page == "scalar-regression":
    st.title("Scalar Regression for Spend Forecast")
    st.write("**Opportunity:** Predict future cloud resource usage trends to inform pricing agreements.")
    st.write("**AI Solution:** I developed and trained a scalar regression model to use historical time-series data for analysis.")
    segment = st.selectbox("Select Customer Segment:", data["Customer_Segment"].unique())
    filtered_data = data[data["Customer_Segment"] == segment]
    st.write(f"### Spend Forecast for {segment} Customers")
    x = list(range(1, 13))
    y = filtered_data["Monthly_Cloud_Spend"].sample(12, random_state=42).sort_values().values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker='o')
    ax.set_title(f"Spend Forecast ({segment})")
    ax.set_xlabel("Month")
    ax.set_ylabel("Projected Spend ($)")
    st.pyplot(fig)

elif page == "vae-anomaly-detection":
    st.title("Variational Autoencoder (VAE) for Anomaly Detection")
    st.write("**Opportunity:** Identify inefficiencies or unusual trends in resource usage to identify savings or optimizations.")
    st.write("**AI Solution:** I developed and trained a VAE to detect anomalies in usage patterns, aiding cost optimization.")
    threshold_multiplier = st.slider("Set Threshold Multiplier (mean + (multiplier x standard deviation; lower = more sensitive/more flagged; higher = less sensitive/fewer flagged (extreme)):", 1.0, 3.0, 1.5)
    threshold = data["Monthly_Cloud_Spend"].mean() + threshold_multiplier * data["Monthly_Cloud_Spend"].std()
    anomalies = data[data["Monthly_Cloud_Spend"] > threshold]
    st.write(f"Detected {len(anomalies)} anomalies (Spend > ${threshold:,.2f}).")
    st.dataframe(anomalies)

elif page == "nlp-customer-insights":
    st.title("Natural Language Processing (NLP) for Customer Insights")
    st.write("**Opportunity:** Extract themes and sentiments from customer feedback to improve product features and services.")
    st.write("**AI Solution:** I developed and trained an NLP model to offer text classification, sentiment analysis, and clustering.")
    user_feedbacks = st.text_area("Enter customer feedback (one per line):", height=150)
    if user_feedbacks:
        feedback_list = user_feedbacks.split("\n")
        results = [{"Feedback": feedback, "Sentiment": analyze_sentiment(feedback)} for feedback in feedback_list if feedback.strip()]
        results_df = pd.DataFrame(results)
        st.write("### Sentiment Analysis Results")
        st.dataframe(results_df)
        # Add horizontal bar graph
    st.write("### Feedback Clustering")
    chart_data = pd.DataFrame({
        'Topic': ['Pricing', 'Usability', 'Support'],
        'Frequency': [120, 90, 60]
    })
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x='Frequency',
        y=alt.Y('Topic', sort='-x'),
        color='Topic'
    ).properties(
        title="Feedback Clustering by Topic"
    )
    st.altair_chart(bar_chart, use_container_width=True)

elif page == "nst-filters":
    st.title("Create Filters with Neural Style Transfer (NST)")
    st.write("**Overview:** BYOF - Bring Your Own Filter! This project demonstrates Neural Style Transfer (NST) by combining the content of a personal photo with the artistic style of a Kandinsky painting. I applied deep learning to create unique, visually engaging filters for images.")
    st.write("**AI Solution:** My NST model uses deep learning to blend the structure of one image with the artistic style of another.")

    # Display the original photo
    st.subheader("Original Photo")
    st.image("Portfolio_Home_Files/photo.jpg", caption="Original Photo", width=400)

    # Display the Kandinsky painting
    st.subheader("Artistic Style Reference")
    st.image("NST/vii.jpg", caption="Kandinsky Painting", width=400)

    # Display the result
    st.subheader("Resulting Image with Neural Style Transfer")
    st.image("NST/result.png", caption="Styled Image", width=400)

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/NST)")


elif page == "cnn-feature-detection":
    st.title("Image Feature Detection with CNNs")
    st.write("**Overview:** This project demonstrates how Convolutional Neural Networks (CNNs) are used to detect features in images, such as edges, patterns, and textures; this can help with image classification or other vision tasks.")
    st.write("**AI Solution:** I applied my CNN model to a lion image to highlight its ability to detect features like edges and textures, so that so that these fundamental patterns can be used to build higher-level understanding for tasks like image classification, object detection, and automated decision-making in real-world AI applications.")

    # Display the original lion image
    st.subheader("Original Image")
    st.image("CNN/lion.jpg", caption="Original Lion Image", width=400)

    # Display the feature maps (Conv2D output)
    st.subheader("Feature Maps from Convolutional Layers (represent extracted spatial features)")
    st.image("CNN/conv2d_18.png", caption="Conv2D Layer 18", use_container_width=True)
    st.image("CNN/conv2d_19.png", caption="Conv2D Layer 19", use_container_width=True)

    # Display the max pooling output
    st.subheader("Feature Maps after Max Pooling Layers (summarize the most prominent features) ")
    st.image("CNN/max_pooling2d_18.png", caption="Max Pooling Layer 18", use_container_width=True)
    st.image("CNN/max_pooling2d_19.png", caption="Max Pooling Layer 19", use_container_width=True)

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/CNN)")


elif page == "sound-analysis":
    st.title("Analyze Sound with Spectrogram Insights")
    st.write("**Overview:** This project demonstrates how audio data can be transformed into visual spectrograms to reveal patterns in frequency and amplitude over time. Spectrograms allow us to analyze sound in a structured, visual format, enabling insights into the characteristics of different audio signals.")
    st.write("**AI Solution:** My model converts audio files into spectrograms to highlight features such as pitch, energy levels, and temporal patterns. These visualizations provide the foundation for tasks like sound classification, speech recognition, and audio anomaly detection, enabling AI to understand and process audio data in real-world applications.")

    # Select a category
    st.subheader("Explore Audio Files by Category")
    category = st.selectbox("Choose a sound category:", ["dog", "eight", "happy"])

    # Audio folder path
    audio_folder = f"audio/{category}/"
    try:
        audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    except FileNotFoundError:
        st.error(f"Audio folder not found: {audio_folder}")
        audio_files = []

    # Select an audio file
    if audio_files:
        selected_file = st.selectbox("Choose an audio file:", audio_files)
        file_path = os.path.join(audio_folder, selected_file)

        # Play the selected audio file
        st.audio(file_path, format="audio/wav")

        # Generate and display the spectrogram
        st.subheader("Spectrogram of the Selected Audio")
        import matplotlib.pyplot as plt
        from scipy.io import wavfile
        import numpy as np

        try:
            sample_rate, audio_data = wavfile.read(file_path)
            plt.figure(figsize=(10, 4))
            plt.specgram(audio_data, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="viridis")
            plt.title(f"Spectrogram of {selected_file}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label="Intensity (dB)")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error generating spectrogram: {e}")
    else:
        st.warning("No audio files found in the selected category.")

    # Add a link to GitHub for the code
    st.write("[View Full Code on GitHub](https://github.com/tm8203/melvin-ai-portfolio/tree/main/Sound)")

# Ensure sidebar collapse logic applies correctly
if st.session_state.get("needs_rerun", False):
    st.session_state.needs_rerun = False  # Reset flag to prevent loops
    st.rerun()  # Safe place to trigger refresh

