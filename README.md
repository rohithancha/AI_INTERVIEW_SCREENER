# AI Interview Screener

A Python script for evaluating interview responses using NLP.

## Features
- Custom evaluation criteria with weighting.
- Command-line interface for input.
- Generates detailed reports for candidate responses.

## Files
- `interview_screener.py`: The basic script for evaluating candidates.
- `interview_screener_v2.py`: Extended version with custom weighting and new criteria.
- `interview_screener_with_cli.py`: Version with a command-line interface for input.
- `interview_screener_detailed_report.py`: Version that generates a detailed breakdown of scores.

## Requirements
- Python 3.x
- `transformers`
- `sklearn`
- `numpy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rohithancha/AI_INTERVIEW_SCREENER.git


cd AI_INTERVIEW_SCREENER
pip install -r requirements.txt
python interview_screener_with_cli.py
