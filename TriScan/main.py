import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load the trained model
learner = load_learner('model.pkl')

st.title("Image Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Check if predict button is clicked
    if st.button('Predict', key='prediction_button', help="Click here to make a prediction"):
        # Make predictions
        is_healthy, _, probs = learner.predict(image)

        # Get the predicted class label
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class_label = learner.dls.vocab[predicted_class_idx]

        # Display prediction
        st.subheader("Prediction:")
        st.write(f"Predicted Class: {predicted_class_label}")
        st.write(f"Probability Score: {probs[predicted_class_idx].item()*100}%")

        # Apply some styling
        st.markdown("---")
        st.success("Prediction complete!")
