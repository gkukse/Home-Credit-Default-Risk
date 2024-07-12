# Start from a base image
FROM python:3.11-slim
RUN sudo apt-get install libgomp1

# Set the working directory
WORKDIR /main

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["model.pkl", "main.py", "./"] .

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]