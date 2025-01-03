FROM python:3.12.8-slim

ENV PATH="/usr/local/bin:${PATH}"

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["python", "src/app.py"]
