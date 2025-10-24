# AI-Accelerated Microfluidic Design Tool

This project is a web-based application built with Streamlit that demonstrates the use of a deep learning model to accelerate the design and analysis of microfluidic devices. Instead of running slow, computationally expensive traditional fluid dynamics (CFD) simulations, this tool uses a pre-trained U-Net model to provide instantaneous predictions of fluid behavior.

## Overview

The application allows a user to interactively design a simple microfluidic channel and immediately visualize the resulting pressure field as predicted by an AI model. This provides a powerful example of how AI can be integrated into engineering workflows for rapid design space exploration and optimization.

## How It Works

1.  **Interactive Design**: A user specifies the geometry of a simple vertical channel using a slider in the web interface. This parameter controls the width of the channel.
2.  **Geometry Generation**: The application generates a 2D grid (`10x20` pixels) representing the channel geometry. In this grid, a value of `1` represents the fluid path (the channel) and `0` represents solid walls.
3.  **AI Prediction**: This geometry grid is fed as an input tensor to a pre-trained Keras/TensorFlow model (`unet_flow_model.h5`).
4.  **Instant Results**: The model, a U-Net architecture, processes the input and instantly outputs a prediction of the steady-state fluid dynamics. The output consists of three channels:
    *   U-velocity field (horizontal velocity component)
    *   V-velocity field (vertical velocity component)
    *   Pressure field
5.  **Visualization**: The application extracts the predicted pressure field and displays it as a 2D contour plot, giving the user immediate visual feedback on their design.

## Features

-   **Real-time Feedback**: Modify the channel width and see the predicted pressure field update in seconds.
-   **AI-Powered Simulation**: Leverages a U-Net model to bypass the need for traditional CFD solvers for rapid analysis.
-   **Optimization Demonstration**: Includes a conceptual "mock optimization" routine that demonstrates how the fast predictions from the model could be used in a loop to automatically search for a design that minimizes a target metric (e.g., pressure drop).
-   **User-Friendly Interface**: Built with Streamlit for a clean and interactive user experience.

## Model Details

-   **Model File**: `unet_flow_model.h5`
-   **Architecture**: U-Net. This is a type of convolutional neural network (CNN) particularly well-suited for image-to-image translation tasks, such as mapping a geometry image to a pressure field image.
-   **Input Shape**: `(batch_size, 10, 20, 1)`. A 10x20 single-channel image representing the geometry.
-   **Output Shape**: `(batch_size, 10, 20, 3)`. A 10x20 three-channel image representing the U-velocity, V-velocity, and Pressure fields.

## How to Run the Application

Follow these steps to run the application on your local machine.

### 1. Prerequisites

Ensure you have Python 3.8+ installed. You will also need the model file `unet_flow_model.h5` placed in the same directory as the application script.

### 2. Installation

Open your terminal or command prompt and install the required Python libraries:

```bash
pip install streamlit tensorflow numpy matplotlib
```

### 3. Execution

Navigate to the project directory (the folder containing `app.py` and `unet_flow_model.h5`) in your terminal and run the following command:

```bash
streamlit run app.py
```

Your default web browser will automatically open a new tab with the running application.

## File Structure

```
flow/
├── app.py              # The main Streamlit application script
├── unet_flow_model.h5  # The pre-trained Keras deep learning model
└── README.md           # This documentation file
```