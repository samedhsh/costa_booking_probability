FROM python:3.10

WORKDIR /app

# Copy app folder into container
COPY app /app

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "Home_Page.py"]
