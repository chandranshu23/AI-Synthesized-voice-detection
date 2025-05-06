FROM python:3.10-slim

#Installing FFMPEG
RUN ap-get update && apt-get install -y ffmpeg

#setting working directory
WORKDIR /app

#copying project files
COPY . .

#installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

#exposing port
EXPOSE 8000

#run fastAPI
CMD [ "uvicorn", "main;app", "--host", "0.0.0.0", "--port", "8000" ]