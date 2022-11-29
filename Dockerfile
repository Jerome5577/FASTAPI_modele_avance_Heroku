# backend/Dockerfile



FROM python:3.9.13


#
WORKDIR /app


RUN apt-get update
RUN apt-get install  -y


# 
COPY requirements.txt .

# 
RUN pip install -r requirements.txt

# 
COPY . .


#
EXPOSE 8080

# 
CMD ["python", "main.py"]