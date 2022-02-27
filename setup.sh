PYTHON_EXEC=python3.8

$PYTHON_EXEC -m venv venv && \
source venv/bin/activate && \
python -m pip install -r requirements.txt && \
echo "Done!"
