# Metaculus Gemini Forecaster

A Metaculus forecasting bot powered by Google's Gemini AI model with built-in search capabilities and structured output formatting.

## Overview

This forecasting bot uses Google's Gemini AI to analyze and predict outcomes on Metaculus questions. It's designed to:

- Generate forecasts for binary, multiple-choice, and numeric questions
- Utilize Gemini's built-in search tools for research
- Provide structured predictions with detailed rationales
- Fall back to alternative LLMs if Gemini encounters issues
- Run automatically via GitHub Actions or locally

## Key Features

- **Gemini-powered forecasting**: Uses Google's advanced generative AI model with structured output schemas
- **Integrated search**: Leverages Gemini's built-in Google Search capabilities for research
- **Fallback capability**: Can use alternative LLMs as backup if Gemini fails
- **Multiple question formats**: Handles binary, multiple-choice, and numeric distribution forecasts
- **JSON-structured responses**: Uses structured schemas for consistent output formatting
- **Configurable parameters**: Adjustable temperature, concurrency limits, and more

## Quick Start with GitHub Actions

1. **Fork the repository**
2. **Set required secrets**:
   - `METACULUS_TOKEN`: Your Metaculus API token
   - `GEMINI_API_KEY`: Your Google AI Gemini API key
   - Optional: API keys for fallback models (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
3. **Enable Actions** and run the workflow

## Getting Required API Keys

### Metaculus Token
1. Go to https://metaculus.com/aib
2. Create a bot account
3. Get your API token from the AIB page

### Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Generate a new API key

## Running Locally

```bash
# Clone repository
git clone https://github.com/yourusername/metaculus-gemini-forecaster.git

# Install dependencies
poetry install

# Create .env file with your API keys

# Run on tournament questions
poetry run python main.py --mode tournament

# Run on test questions
poetry run python main.py --mode test_questions
```

## Configuration Options

```
usage: main.py [-h] [--mode {tournament,quarterly_cup,test_questions}] [--model MODEL]
               [--fallback-model FALLBACK_MODEL] [--fallback-temp FALLBACK_TEMP]
               [--fallback-timeout FALLBACK_TIMEOUT] [--publish | --no-publish]
               [--skip-previous | --no-skip-previous]
               [--predictions-per-question PREDICTIONS_PER_QUESTION]

options:
  --mode {tournament,quarterly_cup,test_questions}  Specify the run mode
  --model MODEL                                     Override GEMINI_MODEL_NAME env var
  --fallback-model FALLBACK_MODEL                   Fallback LLM model name
  --fallback-temp FALLBACK_TEMP                     Temperature for fallback LLM
  --fallback-timeout FALLBACK_TIMEOUT               Timeout for fallback LLM call
  --publish/--no-publish                            Publish predictions to Metaculus
  --skip-previous/--no-skip-previous                Skip questions already forecasted
  --predictions-per-question                        Number of predictions per question
```

## Environment Variables

- `GEMINI_API_KEY`: Required - API key for Google Gemini
- `GEMINI_MODEL_NAME`: Optional - Override default Gemini model
- `METACULUS_TOKEN`: Required - API token for Metaculus
- `OPENAI_API_KEY`: Optional - Only needed if using OpenAI model as fallback
- `ANTHROPIC_API_KEY`: Optional - Only needed if using Anthropic model as fallback

## Technical Details

The bot uses structured output schemas to ensure consistent JSON responses from Gemini:

- Binary questions: Returns a probability between 0 and 1
- Multiple-choice questions: Returns probabilities for each option
- Numeric questions: Returns a distribution with values at specified percentiles

The default Gemini model is "gemini-2.5-pro-preview-03-25" and includes Google Search as a tool by default.

## Example Usage

```bash
# Run on AI tournament with GPT-4 fallback
poetry run python main.py --mode tournament --fallback-model gpt-4o

# Run test questions with Claude fallback and higher temperature
poetry run python main.py --mode test_questions --fallback-model claude-3-opus-20240229 --fallback-temp 0.7
```

## Support

For support with this bot or the Metaculus forecasting tournament, contact `ben@metaculus.com`.