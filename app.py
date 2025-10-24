import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration and Constants ---
MODEL_PATH = 'unet_flow_model.h5'
GRID_HEIGHT = 10  # Match the model's expected input height
GRID_WIDTH = 20   # Match the model's expected input width

# --- Model Loading ---
# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_model():
    """Loads the saved Keras model from the specified path."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the model is in the same directory as app.py.")
        return None
    try:
        # Set compile=False as we only need the model for inference,
        # which avoids issues with deserializing optimizers and metrics.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Geometry and Visualization Functions ---
def generate_geometry(width_param: float) -> np.ndarray:
    """
    Generates a geometry mask based on a width parameter matching the model's input size.
    This creates a simple vertical channel.
    
    Args:
        width_param (float): A normalized parameter (0.1 to 1.0) controlling channel width.

    Returns:
        np.ndarray: A (GRID_HEIGHT, GRID_WIDTH, 1) NumPy array representing the geometry mask.
    """
    # Calculate the pixel width of the channel
    channel_width_pixels = int(GRID_WIDTH * width_param)
    
    # Calculate start and end columns to center the channel
    start_col = (GRID_WIDTH - channel_width_pixels) // 2
    end_col = start_col + channel_width_pixels
    
    # Create the geometry mask (0 for solid, 1 for fluid)
    geometry_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    geometry_mask[:, start_col:end_col] = 1.0
    
    # Reshape to (128, 128, 1) for the model
    return np.expand_dims(geometry_mask, axis=-1)

def visualize_pressure(pressure_field: np.ndarray):
    """
    Generates and displays a 2D contour plot of the pressure field.
    
    Args:
        pressure_field (np.ndarray): A (128, 128) NumPy array of pressure values.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    contour = ax.contourf(pressure_field, cmap='viridis', levels=50)
    fig.colorbar(contour, ax=ax, label='Pressure')
    ax.set_title("Predicted Pressure Field")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_aspect('equal')
    st.pyplot(fig)
    plt.close(fig) # Explicitly close the figure to free up memory

# --- Main Streamlit Application ---
def main():
    st.set_page_config(layout="wide")
    st.title("AI-Accelerated Microfluidic Design Tool")

    # Load the model with a user-friendly spinner
    with st.spinner('Loading the AI model, please wait...'):
        model = load_model()

    if model is None:
        st.stop() # Stop execution if the model failed to load

    st.success("AI Model loaded successfully!")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Input Parameters")
    width_param = st.sidebar.slider(
        'Channel Width Parameter', 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )

    # --- Main Panel for Prediction and Visualization ---
    st.header("Single Design Prediction")
    
    if st.button("Predict Flow Field"):
        with st.spinner("Generating geometry and running prediction..."):
            # 1. Generate the input geometry tensor
            geometry_mask = generate_geometry(width_param)
            # Add batch dimension for the model: (1, 128, 128, 1)
            input_tensor = np.expand_dims(geometry_mask, axis=0)

            # 2. Use the model to predict the output (calling the model is faster for single inference)
            predicted_tensor = model(input_tensor, training=False)

            # 3. Separate the predicted fields (U, V, P)
            # Output shape is (1, 128, 128, 3)
            pressure_field = predicted_tensor[0, :, :, 2] # Pressure is the 3rd channel (index 2)

            # 4. Display the visualization
            st.subheader("Prediction Results")
            visualize_pressure(pressure_field)

    # --- Optimization Component (Conceptual) ---
    st.header("Optimization Demonstration")
    st.info("This is a simplified loop to demonstrate how the model can be used for optimization.")

    if st.button("Run Mock Optimization Search (5 Iterations)"):
        st.subheader("Optimization Log")
        progress_bar = st.progress(0)
        results_placeholder = st.empty()
        results_text = ""

        for i in range(5):
            with st.spinner(f"Running iteration {i+1}/5..."):
                # Generate a random width for this iteration
                random_width = np.random.uniform(0.1, 1.0)
                
                # Generate geometry and predict
                geometry_mask = generate_geometry(random_width)
                input_tensor = np.expand_dims(geometry_mask, axis=0)
                predicted_tensor = model(input_tensor, training=False)
                pressure_field = predicted_tensor[0, :, :, 2]
                
                # Calculate a mock 'Pressure Drop' metric
                # A more realistic metric would be inlet_pressure_avg - outlet_pressure_avg
                mock_pressure_drop = np.mean(pressure_field)
                
                # Display the result for the current iteration
                log_entry = f"Iteration {i+1}: Width Param = {random_width:.3f} -> Mock Pressure Drop = {mock_pressure_drop:.4f}\n"
                results_text += log_entry
                results_placeholder.code(results_text, language='text')
                
                # Update progress bar
                progress_bar.progress((i + 1) / 5)
        
        st.success("Mock optimization search complete!")

if __name__ == '__main__':
    main()
