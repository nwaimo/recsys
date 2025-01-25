# Movie Recommender System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A sophisticated recommendation engine that provides personalized movie suggestions using collaborative filtering, content-based filtering, and hybrid approaches.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Multiple Recommendation Methods**:
  - Collaborative filtering with ANN (Approximate Nearest Neighbors)
  - Content-based filtering using movie metadata
  - Hybrid recommendations combining multiple approaches

- **Real-time Processing**:
  - Efficient batch processing of new ratings
  - Automatic model updates
  - Caching system for fast responses

- **User Experience**:
  - User and admin interfaces
  - Personalized recommendations
  - Similar movie suggestions
  - Genre-based filtering

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/nwaimo/RecSys.git
cd RecSys
```

2. **Create a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

Run the system:
```bash
python run.py
```

## 💻 Usage

### User Mode
```bash
# Start the system in user mode
python run.py

# Available commands:
1. Search for movies
2. Get personalized recommendations
3. Rate movies
4. View popular movies
5. Switch to admin mode
```

### Admin Mode
```bash
# Access admin features with password
1. System status
2. Save model
3. Evaluate model
4. Clear cache
5. Switch to user mode
```

## 🏗 Architecture

```
RecSys/
├── src/
│   ├── recommender.py    # Core recommendation engine
│   ├── data_loader.py    # Data management
│   ├── evaluation.py     # System evaluation
│   ├── batch_processor.py # Real-time processing
│   └── main.py          # CLI interface
├── tests/               # Unit and integration tests
├── requirements.txt     # Dependencies
└── README.md           # Documentation
```

## 🔍 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - [@nwaimo](https://github.com/nwaimo)

Project Link: [https://github.com/nwaimo/RecSys](https://github.com/nwaimo/RecSys)
