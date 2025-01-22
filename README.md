# Movie Recommender System

A sophisticated movie recommendation system built with Python, featuring collaborative filtering, content-based filtering, and hybrid recommendations. The system includes real-time processing, analytics monitoring, and both user and admin interfaces.

## Features

- Collaborative filtering using Approximate Nearest Neighbors (ANN)
- Content-based filtering using movie genres and TF-IDF
- Hybrid recommendation system
- Real-time rating updates with batch processing
- System monitoring and analytics
- User and admin interfaces
- Performance evaluation metrics

## Project Structure

```
RecSys/
├── src/
│   ├── __init__.py
│   ├── analytics.py      # Analytics monitoring
│   ├── batch_processor.py # Batch processing functionality
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── main.py          # Entry point and CLI interface
│   ├── recommender.py   # Core recommendation logic
│   └── utils.py         # Utility functions
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RecSys.git
cd RecSys
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the recommender system:
```bash
python -m src.main
```

### User Menu Options

1. Get similar movies
2. Get personalized recommendations
3. Get genre recommendations
4. Add new rating
5. Switch to admin mode
6. Exit

### Admin Menu Options (Password: admin123)

1. View analytics dashboard
2. View analytics plots
3. Save analytics
4. View batch status
5. Evaluate system
6. Save model
7. Switch to user mode
8. Exit

## Data

The system uses the MovieLens 1M dataset, which includes:
- 1 million ratings
- 6,000 users
- 4,000 movies
- Rating scale: 1-5 stars

The dataset will be automatically downloaded on first run.

## Components

### Analytics Monitor
- Tracks system metrics
- Monitors user activity
- Records response times
- Generates performance reports

### Batch Processor
- Handles real-time rating updates
- Processes ratings in batches
- Updates recommendation models
- Maintains system performance

### Data Loader
- Downloads and manages dataset
- Preprocesses movie and user data
- Creates feature matrices
- Handles data transformations

### Recommender System
- Implements core recommendation logic
- Manages user preferences
- Generates personalized recommendations
- Maintains recommendation quality

## Performance Metrics

The system evaluates recommendations using:
- Precision
- Recall
- F1 Score
- Diversity
- Novelty
- Coverage
- Serendipity

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
