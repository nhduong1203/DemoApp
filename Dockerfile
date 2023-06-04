FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip
RUN pip3 install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --upgrade -r docker-requirements.txt


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

#docker build -t transfer_app .
#docker run --name test -p 8501:8501 transfer_app
#http://localhost:8501/