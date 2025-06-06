# ⚽ Fantasy Premier League Assistant

An intelligent platform combining machine learning and optimization algorithms to help you **dominate your Fantasy Premier League**.

---

## 🚀 Features

- **Real-time Player Analytics**
- **ML-Powered Performance Predictions** (Random Forest)
- **Squad Optimizer** (Linear Programming)
- **Gameweek Planner**
- **Team Saving** (MongoDB Integration)
- **Responsive Web Interface**

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- [Make](https://www.gnu.org/software/make/) (optional, but recommended)
- [Docker](https://www.docker.com/) (optional, for container deployment)

---

### Clone the Repository

git clone https://github.com/dalibouzir/MLops.git
cd MLops

---

### Using Make (Recommended)

make install      # Sets up virtual environment & installs dependencies

---

### Manual Installation

python -m venv venv

# On Windows
.\venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

---

## 🖥️ Usage

### Start the Application

make run               # OR: flask run (if Flask)
# OR: uvicorn FastAPI:app --reload   (if using FastAPI)

Visit http://localhost:5000 (for Flask)  
or http://localhost:8000 (for FastAPI)

---

### Makefile Commands

Command         | Description
----------------|---------------------------------------
make install    | Setup virtualenv + dependencies
make run        | Launch development server
make test       | Run pytest suite
make lint       | Check code quality with flake8
make format     | Auto-format with Black
make clean      | Remove virtualenv and cache files

---

## 🐛 Troubleshooting

**Makefile not working on Windows?**

You can install `make` via Chocolatey:

# Install Chocolatey (if not already installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; `
  [System.Net.ServicePointManager]::SecurityProtocol = `
  [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
  iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Then install make
choco install make

For questions, open an issue or contact the maintainer: [dalibouzir](https://github.com/dalibouzir)

---

## 🐳 Docker Compose (Optional)

To run the app and dependencies (e.g. MongoDB) via Docker Compose:

docker-compose up --build

Edit `docker-compose.yml` to configure ports, MongoDB, etc.

---

## 📊 Kibana (Optional)

If using the ELK stack, Kibana UI will be at:
http://localhost:5601

---

## 🚀 MLflow (Optional)

To launch MLflow UI for experiment tracking:

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

Open http://localhost:5001 in your browser.

---

## 💬 Questions & Contributions

- Open an issue for help or suggestions
- Contact maintainer: dalibouzir
- **PRs are welcome!**

---

> **Copy ALL the content above in one go** — it’s all a single markdown block, nothing separated!  
> Let me know if you want to add more integrations or documentation.
