# Lmah_Project

Public website: https://lmahdeploy-cdf8j98whwpujfjchxqdx7.streamlit.app/

# Lmah_Project
# Capstone README

Welcome to the Capstone Project, where we developed an intelligent traffic management system with two primary objectives:
1. Detect emergency vehicles and dynamically adjust traffic signals to prioritize their passage.
2. Optimize overall traffic flow by using reinforcement learning to synchronize traffic lights, improving throughput and reducing congestion (Green Wave).

The project consists of three parts: detecting emergency vehicles with YOLOv8, optimizing traffic signals with reinforcement learning, and deploying these models via a public-facing web app for users to interact with and test.

---

## Project 1: **YOLOv8 Emergency Vehicle Detection**
### File: `Working_version_of_capstone.ipynb`

This project uses a YOLOv8 (You Only Look Once) model to detect emergency vehicles in real-time from video footage. The primary objective is to identify emergency vehicles (e.g., ambulances, fire trucks, etc.) with active sirens and automatically open the traffic signal for them, enabling uninterrupted passage through intersections.

### Key Features:
- **Model**: The YOLOv8 model is trained to detect emergency vehicles with high accuracy.
- **Dynamic Signal Control**: Upon detecting an emergency vehicle, the system automatically overrides the current traffic signal pattern to give the vehicle a green light.
- **Real-Time Processing**: The system operates in real-time, processing video feeds and making immediate decisions.
- **Data Preprocessing**: Preprocessed video data for training the model includes labeled bounding boxes and classes for emergency vehicles.
- **Visualization**: Bounding boxes are drawn on the detected vehicles in the real-time video feed, and the signal actions are logged.

### Results:
- The system successfully identified emergency vehicles in various traffic scenarios with high accuracy, leading to a significant reduction in response time for emergency vehicles navigating busy intersections.

---

## Project 2: **Reinforcement Learning for Traffic Optimization**
### File: `sumo-rl-main`

This project focuses on using reinforcement learning (RL) to dynamically optimize the flow of traffic lights in an urban environment, particularly through the implementation of the **Green Wave Algorithm**. The RL agent learns to control signal timings based on traffic conditions to improve overall traffic flow and reduce congestion.

### Key Features:
- **Green Wave Algorithm**: The RL agent is trained to implement a Green Wave system, where groups of cars move through multiple intersections without stopping, significantly reducing congestion.
- **Environment**: The project uses SUMO (Simulation of Urban MObility), a traffic simulation tool, to simulate real-world traffic scenarios.
- **Reinforcement Learning**: A Deep Q-Network (DQN) is used to train the agent to maximize traffic flow by dynamically adjusting signal timings.
- **Dynamic Signal Timing**: The agent learns from traffic conditions in real-time, adjusting signals at each intersection to minimize waiting times and maximize throughput.
- **Reward System**: The RL system is guided by a reward function based on reducing waiting times and increasing the number of vehicles passing smoothly.

### Results:
- **Before**: Traffic simulations without RL showed high congestion and inefficient signal timings, especially during peak hours.
- **After**: The RL model improved the traffic flow by reducing waiting times, decreasing stop-and-go behavior, and applying the Green Wave successfully across multiple intersections. Comparison metrics showed a significant reduction in congestion after the RL implementation.

---

## Project 3: **Deployment and User Interaction**
### File: `StreamLit_deployment`

This final project combines the two previous projects and deploys them on a publicly accessible website where users can interact with the models and test their performance in different traffic scenarios.

### Key Features:
- **Web App Interface**: Built using Streamlit, the website allows users to interact with the models and view their performance.
- **Model Demonstration**: Users can upload video files or access live traffic camera feeds to see how the YOLOv8 emergency vehicle detection model works.
- **Interactive Traffic Simulation**: Users can simulate traffic conditions using SUMO and watch how the RL agent optimizes the traffic flow in real-time.
- **Visualization**: Real-time visualizations show the detection of emergency vehicles, traffic flow improvements, and signal timing adjustments.
- **Comparison Tool**: Users can compare the traffic conditions before and after applying the RL agent (Green Wave) to understand the impact of the optimization.

### Results:
- **Before vs. After Comparison**: The platform presents comparison metrics showing the improvements in traffic flow after applying the YOLOv8 model and the RL-based Green Wave. Metrics such as average waiting time, vehicle throughput, and trip times are visualized for easy understanding.
- **User Interaction**: The website provides an intuitive interface for users to upload their own data, test the models, and visualize the results in real-time.

---

## Data Preprocessing and Visualization Codes

All three projects include various data preprocessing and visualization steps:
- **Data Preprocessing**: Includes cleaning, labeling, and transforming the video and traffic data for model training.
- **Visualization**: Custom visualizations of model predictions (for YOLOv8) and traffic simulations (for the RL-based Green Wave system).
- **Performance Metrics**: Graphs and tables compare key metrics such as waiting times, vehicle throughput, trip times, and detection accuracy.

---

## Results Summary:
- **Emergency Vehicle Detection**: Successfully reduced response time for emergency vehicles by dynamically opening traffic signals.
- **Reinforcement Learning Optimization**: Improved traffic flow and reduced congestion, with significant performance improvements in simulation results after RL-based Green Wave implementation.
- **Deployment**: A user-friendly platform that allows users to test, visualize, and compare the model performances.

---

### How to Run:
1. **YOLOv8 Emergency Vehicle Detection**: Open `Working_version_of_capstone.ipynb` in a Jupyter Notebook and run the code. Ensure the necessary video files and labeled datasets are available.
2. **Reinforcement Learning for Traffic Optimization**: Run the SUMO traffic simulation by executing `sumo-rl-main` with the pre-configured SUMO environment. Ensure the necessary files and models are in place.
3. **StreamLit Deployment**: Run the `StreamLit_deployment` app to launch the public-facing web application. Use the Streamlit command: `streamlit run StreamLit_deployment.py` to start the server.

---

### Requirements:
- Python 3.x
- Torch
- SUMO
- Streamlit
- YOLOv8 dependencies (OpenCV, Numpy, PyTorch)
- Reinforcement Learning dependencies (TensorFlow, SUMO, DQN)

---

Feel free to explore each project in more detail, run the simulations, and test the models!
