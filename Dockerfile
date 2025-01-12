FROM python:3.12.8-slim

ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV FLASK_APP=src/app.py
ENV FLASK_ENV=production

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
