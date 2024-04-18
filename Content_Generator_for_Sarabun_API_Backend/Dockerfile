FROM python:3.11

WORKDIR /app

# Install Poppler
RUN apt-get update \
  && apt-get install poppler-utils -y

# Install Tesseract
ENV PYHTONUNBUFFERED=1
RUN apt-get update \
  && apt-get -y install tesseract-ocr-tha \
  && apt-get -y install tesseract-ocr

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Expose the port the app will run on
EXPOSE 8150

CMD ["python", "backend_API.py"]