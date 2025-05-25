.PHONY: install run test clean format lint security

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# Create venv and install dependencies
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run your main program (edit the filename if needed)
run:
	$(PYTHON) main.py

# Run tests (pytest should be in requirements.txt)
test:
	$(PYTHON) -m pytest tests/

# Remove __pycache__, .pyc files, and venv for a fresh setup
clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Black code formatter (autopep8/black in requirements)
format:
	$(PYTHON) -m black .

# Lint with flake8 (should be in requirements)
lint:
	$(PYTHON) -m flake8 .

# Security check with bandit (add bandit to requirements)
security:
	$(PYTHON) -m bandit -r .
