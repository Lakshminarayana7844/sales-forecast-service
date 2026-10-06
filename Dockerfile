FROM python:3.12-slim
WORKDIR /srv
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn joblib
COPY forecast ./forecast
COPY data ./data
RUN python -m forecast --out models
EXPOSE 8000
CMD ["uvicorn", "forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
