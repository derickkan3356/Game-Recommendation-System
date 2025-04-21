# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This assumes app.py and prediction_pipeline.py are in the same directory as the Dockerfile
COPY . .

# Make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

# Define environment variable for Streamlit Cloud Run deployment
# This helps Streamlit configure itself correctly for the Cloud Run environment
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Run app.py when the container launches using Streamlit
# Use --server.port $PORT because Cloud Run expects the app to listen on the port specified by the PORT env var (usually 8080)
# Use --server.headless true as Cloud Run doesn't have a browser GUI.
# CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true"]

CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.headless", "true", "--server.fileWatcherType", "none"]