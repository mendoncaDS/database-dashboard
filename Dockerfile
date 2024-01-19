# Use Python 3.10 as base image
FROM python:3.10

# Set the working directory in the Docker image
WORKDIR /usr/src/app

# Copy the requirements file into the image
COPY requirements.txt .

# Install dependencies using pip
RUN pip install -r requirements.txt

RUN pip install streamlit

# Copy the Streamlit script into the image
COPY database_dashboard.py .
COPY render.py .
COPY auxiliary.py .
COPY constants.py .
COPY NGUSTRAT.png .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the entry point to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "database_dashboard.py", "--server.port=8501"]
