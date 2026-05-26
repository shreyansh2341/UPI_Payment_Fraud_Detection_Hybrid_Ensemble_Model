# Use official Python 3.10 image as the base
FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces requires running Docker as a non-root user
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory and add local bin to PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the backend requirements file first to leverage Docker cache
COPY --chown=user backend/requirements.txt ./backend/requirements.txt

# Install the Python dependencies
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Copy the rest of the project files, including models and data directories
# We copy everything because backend/app.py depends on src/, models/, etc.
COPY --chown=user . .

# Ensure the 'data' directory exists so SQLite can write to it
RUN mkdir -p data && chown -R user:user data

# Expose the default port used by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI application using uvicorn
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
