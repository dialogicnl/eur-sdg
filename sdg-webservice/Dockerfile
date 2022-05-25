FROM python:3.6
WORKDIR /app
COPY ./requirements.txt .
RUN pip3 install --no-cache-dir -r ./requirements.txt
RUN groupadd -g 999 appuser && useradd -r -u 1001 -g appuser appuser
RUN chown -R appuser:appuser /app
RUN mkdir /home/appuser && chown appuser:appuser /home/appuser
USER appuser
COPY . .
EXPOSE 5000
CMD ["uwsgi", "--socket=0.0.0.0:5000", "--buffer-size=32768","--protocol=http", "--pythonpath=/app", "-w", "main:app"]
HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1
