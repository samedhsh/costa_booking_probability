FROM python:3.8-slim

WORKDIR /

# Copy app folder into container
COPY app /

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "HomePage.py"]
