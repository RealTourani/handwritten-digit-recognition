FROM tensorflow/tensorflow:latest
WORKDIR /app
COPY detect.py trained_model.h5 /app/
RUN pip install Pillow
RUN pip install tensorflow

CMD ["python", "detect.py"]
