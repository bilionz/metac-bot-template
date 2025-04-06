# main_with_no_framework.py

import asyncio
import datetime
import json
import os
import re
import logging
from typing import Literal, Optional, Dict, Any, Tuple, List, Union
import argparse
import dotenv
dotenv.load_dotenv()

import numpy as np
import requests

# --- Google Generative AI Imports ---
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

# --- Logger Setup ---
# Basic logging, can be configured further if needed
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google.api_core").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


######################### CONSTANTS #########################
# --- Metaculus Config ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
API_BASE_URL = "https://www.metaculus.com/api"
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}

# --- Gemini Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-03-25") # Use 1.5 Pro as default

# --- Bot Behavior Config ---
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
# NUM_RUNS_PER_QUESTION = 1 # Keep single run as per original main.py logic
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True
SAVE_REPORTS = True # Flag to save detailed rationale locally
REPORTS_FOLDER = "gemini_forecast_reports_no_framework"

# --- Tournament/Question IDs ---
# Use IDs from the template or update as needed
Q1_2025_AI_BENCHMARKING_ID = 32627
TOURNAMENT_ID = Q1_2025_AI_BENCHMARKING_ID # Default tournament
EXAMPLE_QUESTIONS = [
    (578, 578),  # Human Extinction - Binary
    (14333, 14333),  # Age of Oldest Human - Numeric
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice
]

# --- Concurrency ---
CONCURRENT_REQUESTS_LIMIT = 2 # Limit concurrent calls to Gemini API
gemini_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


# --- Define Gemini Response Schemas (Unchanged from previous versions) ---
# Schema for Binary Questions
binary_prediction_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'Prediction': types.Schema(type=types.Type.NUMBER, description="Probability (0.0 to 1.0) for the 'Yes' outcome."),
        'Rationale': types.Schema(type=types.Type.STRING, description="Detailed reasoning for the prediction, including analysis of historical data, base rates, search results, and scenarios."),
        'HistoricalData': types.Schema(type=types.Type.STRING, description="Summary of relevant historical data or base rates found or calculated."),
    },
    required=['Prediction', 'Rationale', 'HistoricalData']
)
# Schema for Multiple Choice Questions
multiple_choice_prediction_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'Prediction': types.Schema(
            type=types.Type.ARRAY,
            description="List of options with their assigned probabilities.",
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    'option': types.Schema(type=types.Type.STRING, description="The specific choice option."),
                    'probability': types.Schema(type=types.Type.NUMBER, description="Probability (0.0 to 1.0) assigned to this option. Probabilities should sum to 1.0.")
                },
                required=['option', 'probability']
            )
        ),
        'Rationale': types.Schema(type=types.Type.STRING, description="Detailed reasoning for the probability distribution, including analysis of historical data, base rates, search results, and scenarios for each option."),
        'HistoricalData': types.Schema(type=types.Type.STRING, description="Summary of relevant historical data or base rates found or calculated."),
    },
    required=['Prediction', 'Rationale', 'HistoricalData']
)
# Schema for Numeric Questions
numeric_prediction_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        'Prediction': types.Schema(
            type=types.Type.OBJECT,
            description="Predicted values at specific percentiles.",
            properties={
                'p10': types.Schema(type=types.Type.NUMBER, description="10th percentile value."),
                'p20': types.Schema(type=types.Type.NUMBER, description="20th percentile value."),
                'p40': types.Schema(type=types.Type.NUMBER, description="40th percentile value."),
                'p60': types.Schema(type=types.Type.NUMBER, description="60th percentile value."),
                'p80': types.Schema(type=types.Type.NUMBER, description="80th percentile value."),
                'p90': types.Schema(type=types.Type.NUMBER, description="90th percentile value."),
            },
            required=['p10', 'p20', 'p40', 'p60', 'p80', 'p90']
        ),
        'Rationale': types.Schema(type=types.Type.STRING, description="Detailed reasoning for the percentile distribution, including analysis of historical data, base rates, trends, search results, and scenarios for low/high outcomes."),
        'HistoricalData': types.Schema(type=types.Type.STRING, description="Summary of relevant historical data, base rates or trends found or calculated."),
    },
    required=['Prediction', 'Rationale', 'HistoricalData']
)


######################### HELPER FUNCTIONS (Metaculus Interaction) #########################

def post_question_comment(post_id: int, comment_text: str) -> None:
    """Post a comment on the question page as the bot user."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/comments/create/",
            json={
                "text": comment_text, "parent": None, "included_forecast": True,
                "is_private": True, "on_post": post_id,
            },
            **AUTH_HEADERS, # type: ignore
        )
        response.raise_for_status()
        logger.info(f"Posted comment to post {post_id}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post comment to post {post_id}: {e}")
        # Optional: re-raise or handle differently

def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """Post a forecast on a question."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/questions/forecast/",
            json=[{"question": question_id, **forecast_payload}],
            **AUTH_HEADERS, # type: ignore
        )
        logger.info(f"Prediction Post status code for Q {question_id}: {response.status_code}")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to post prediction for Q {question_id}: {e}")
        # Optional: re-raise or handle differently

def create_forecast_payload(forecast: Union[float, Dict[str, float], List[float]], question_type: str) -> dict:
    """Creates the API payload for Metaculus based on forecast type."""
    if question_type == "binary":
        return {"probability_yes": forecast, "probability_yes_per_category": None, "continuous_cdf": None}
    if question_type == "multiple_choice":
        # Ensure forecast is Dict[str, float]
        if not isinstance(forecast, dict):
             logger.error(f"Invalid forecast type for multiple_choice: {type(forecast)}. Expected dict.")
             # Return a payload indicating failure or default, or raise error
             return {"probability_yes": None, "probability_yes_per_category": {}, "continuous_cdf": None} # Example: empty dict
        return {"probability_yes": None, "probability_yes_per_category": forecast, "continuous_cdf": None}
    if question_type == "numeric":
         # Ensure forecast is List[float] (CDF)
        if not isinstance(forecast, list) or len(forecast) != 201:
             logger.error(f"Invalid forecast type/length for numeric: {type(forecast)}, len={len(forecast) if isinstance(forecast, list) else 'N/A'}. Expected list of 201 floats.")
             # Return a payload indicating failure or default, or raise error
             return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": [0.5]*201} # Example: dummy CDF
        return {"probability_yes": None, "probability_yes_per_category": None, "continuous_cdf": forecast}
    logger.error(f"Unknown question type '{question_type}' for payload creation.")
    return {} # Return empty dict for unknown types


def list_posts_from_tournament(tournament_id: int = TOURNAMENT_ID, offset: int = 0, count: int = 50) -> dict:
    """List posts from the specified tournament."""
    url_qparams = {
        "limit": count, "offset": offset, "order_by": "-hotness",
        "forecast_type": "binary,multiple_choice,numeric",
        "tournaments": [tournament_id], "statuses": "open", "include_description": "true",
    }
    try:
        response = requests.get(f"{API_BASE_URL}/posts/", **AUTH_HEADERS, params=url_qparams) # type: ignore
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list posts for tournament {tournament_id}: {e}")
        return {"results": []} # Return empty structure on failure

def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    """Fetches open question IDs and their post IDs from the default tournament."""
    data = list_posts_from_tournament()
    open_question_id_post_id = []
    post_dict = {}
    for post in data.get("results", []):
        if question := post.get("question"):
             post_dict[post["id"]] = [question]
        elif questions := post.get("questions"): # Handle multi-question posts if needed
            post_dict[post["id"]] = questions

    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                q_id = question.get("id")
                q_title = question.get("title", "No Title")
                q_close = question.get("scheduled_close_time", "N/A")
                if q_id:
                    logger.info(f"Found Open Q: ID={q_id}, Post={post_id}, Closes={q_close}, Title={q_title[:50]}...")
                    open_question_id_post_id.append((q_id, post_id))
                else:
                    logger.warning(f"Found open question in post {post_id} without an ID.")

    logger.info(f"Found {len(open_question_id_post_id)} open questions in tournament {TOURNAMENT_ID}.")
    return open_question_id_post_id

def get_post_details(post_id: int) -> Optional[dict]:
    """Get all details about a post from the Metaculus API."""
    url = f"{API_BASE_URL}/posts/{post_id}/"
    logger.info(f"Getting details for post {post_id} ({url})")
    try:
        response = requests.get(url, **AUTH_HEADERS) # type: ignore
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get details for post {post_id}: {e}")
        return None

def forecast_is_already_made(post_details: dict) -> bool:
    """Check if a forecast has already been made by the authenticated user."""
    # Navigate safely through the nested structure
    try:
        # Check if 'my_forecasts' exists and is not None, then check 'latest'
        latest_forecast = post_details.get("question", {}).get("my_forecasts", {}).get("latest")
        # Check if 'forecast_values' exists within 'latest' and is not None
        if latest_forecast and latest_forecast.get("forecast_values") is not None:
             return True
    except Exception as e: # Catch any unexpected attribute errors etc.
        logger.warning(f"Error checking existing forecast for post {post_details.get('id', 'N/A')}: {e}")
    return False

def clean_indents(text: str) -> str:
     """Removes common leading indentation from multiline strings."""
     return re.sub(r"^\s+", "", text, flags=re.MULTILINE)

def save_report(post_id: int, question_id: int, report_content: str):
    """Saves the detailed rationale to a local file."""
    if not SAVE_REPORTS:
        return
    try:
        if not os.path.exists(REPORTS_FOLDER):
            os.makedirs(REPORTS_FOLDER)
        filename = f"{REPORTS_FOLDER}/report_post{post_id}_q{question_id}_{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Saved report to {filename}")
    except OSError as e:
        logger.error(f"Failed to save report for Q {question_id}: {e}")


######################### GEMINI CALL FUNCTION #########################
async def call_gemini(prompt: str, response_schema: types.Schema, model_name: str = DEFAULT_GEMINI_MODEL) -> Optional[Dict[str, Any]]:
    """Calls the Gemini API asynchronously with schema and search tool."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set. Cannot call Gemini.")
        return None
    
    try:
                    # Initialize client using genai.Client(api_key=...)
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized using API Key.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client with API key: {e}", exc_info=True)
        gemini_client = None # Ensure client is None on failure

    try:
        contents = [types.Content(role="user", parts=[prompt])]
        # Define tools (Search)
        tools = [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=16300,
            response_mime_type="application/json", # MUST request JSON
            response_schema=response_schema, # Pass the specific schema
        )

        response = await asyncio.to_thread(
            gemini_client.models.generate_content, # Calling method on client.models
            model=model_name, # Pass model name here
            contents=contents,
            generation_config=generate_content_config,
            tools=tools,
            stream=False # Ensure non-streaming (default)
        )

        # --- Process Response (same logic as before) ---
        if not response.candidates or not response.candidates[0].content.parts:
            # Check finish reason if available (might indicate blocked content etc.)
            finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
            safety_ratings = getattr(response.candidates[0], 'safety_ratings', [])
            logger.warning(f"Gemini API returned empty or incomplete response. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}")
            # Log the prompt hash or a snippet for debugging difficult cases if needed
            # import hashlib
            # prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            # logger.debug(f"Prompt Hash leading to empty response: {prompt_hash}")
            return None
        
        response_text = response.text
        logger.debug(f"Raw response text received from Gemini:\n{response_text}")
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to parse Gemini JSON response: {json_err}. Response: {response_text}")
            return None
        logger.info(f"Gemini structured response received for schema {response_schema.type}:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")
        # Check against the 'required' fields specified in the schema object
        required_keys = getattr(response_schema, 'required', [])
        if required_keys and not all(key in response_data for key in required_keys):
            missing_keys = [key for key in required_keys if key not in response_data]
            logger.error(f"Gemini response missing required keys: {missing_keys}. Response: {response_data}")
            return None
        return response_data # Success

    # Use the imported google_exceptions if available
    except google_exceptions.GoogleAPIError as api_err:
        logger.error(f"Gemini API error: {api_err}")
        # Specific handling for common errors if needed
        if isinstance(api_err, google_exceptions.InvalidArgument) and "response_schema" in str(api_err):
             logger.error("Error likely related to schema definition or model's compatibility with it.")
        elif isinstance(api_err, google_exceptions.ResourceExhausted):
             logger.error("Quota exceeded. Consider slowing down requests or increasing quota.")
        return None
    except AttributeError as attr_err:
         # This might catch issues if client.models.generate_content doesn't exist
         logger.error(f"AttributeError during Gemini call (check API structure/client init): {attr_err}", exc_info=True)
         return None
    except Exception as e:
         # Catch the original TypeError here as well if it persists
         if isinstance(e, TypeError) and "Part.from_text()" in str(e):
              logger.error(f"Persistent TypeError calling Part.from_text(): {e}", exc_info=True)
         else:
              logger.error(f"Unexpected error during Gemini call: {e}", exc_info=True)
              
         return None


####################### NUMERIC CDF GENERATION ###############

def generate_continuous_cdf(
    percentile_values: dict, # Expects {10: val, 20: val, ... 90: val}
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: Optional[float], # Can be None if open
    lower_bound: Optional[float], # Can be None if open
    zero_point: Optional[float],
) -> list[float]:
    """
    Generates a 201-point CDF list from percentile values and bounds.
    Adapted from template, requires careful handling of bounds.
    """
    if not percentile_values or not all(isinstance(v, (int, float)) for v in percentile_values.values()):
        logger.error(f"Invalid percentile_values for CDF generation: {percentile_values}")
        # Return a default/dummy CDF
        return [0.5] * 201 # Example: flat 0.5 CDF

    # Ensure integer keys and sort
    try:
        int_key_percentiles = {int(k): float(v) for k, v in percentile_values.items()}
        if not all(k in int_key_percentiles for k in [10, 20, 40, 60, 80, 90]):
             raise ValueError("Missing required percentiles")
        sorted_percentiles = dict(sorted(int_key_percentiles.items()))
    except (ValueError, TypeError) as e:
         logger.error(f"Error processing percentile keys/values: {e}. Input: {percentile_values}")
         return [0.5] * 201

    # --- Determine effective range min/max for interpolation ---
    p_min_val = sorted_percentiles[10]
    p_max_val = sorted_percentiles[90]

    # Use bounds if defined, otherwise extrapolate slightly from percentiles
    range_min_eff = float(lower_bound) if not open_lower_bound and lower_bound is not None else p_min_val - abs(p_min_val * 0.1) if p_min_val != 0 else -1.0
    range_max_eff = float(upper_bound) if not open_upper_bound and upper_bound is not None else p_max_val + abs(p_max_val * 0.1) if p_max_val != 0 else 1.0

    # Ensure min < max
    if range_max_eff <= range_min_eff:
        logger.warning(f"Calculated effective range max <= min ({range_max_eff} <= {range_min_eff}). Adjusting.")
        range_max_eff = range_min_eff + 1.0 # Add a small delta

    # --- Prepare points for interpolation (percentile -> value) ---
    # Normalize percentile keys to 0-1 range
    points_for_interp = {float(p) / 100.0: val for p, val in sorted_percentiles.items()}

    # Add endpoints based on bounds
    points_for_interp[0.0] = range_min_eff # Use effective min for 0th percentile
    points_for_interp[1.0] = range_max_eff # Use effective max for 100th percentile

    # Ensure bounds are respected if they were provided
    if not open_lower_bound and lower_bound is not None:
        points_for_interp[0.0] = max(points_for_interp[0.0], float(lower_bound))
        # Clamp other points if they fell below lower_bound
        for p in points_for_interp:
             points_for_interp[p] = max(points_for_interp[p], float(lower_bound))

    if not open_upper_bound and upper_bound is not None:
        points_for_interp[1.0] = min(points_for_interp[1.0], float(upper_bound))
        # Clamp other points if they exceeded upper_bound
        for p in points_for_interp:
            points_for_interp[p] = min(points_for_interp[p], float(upper_bound))

    # Ensure monotonicity after clamping (simple forward pass check)
    sorted_p_keys = sorted(points_for_interp.keys())
    for i in range(len(sorted_p_keys) - 1):
        p0, p1 = sorted_p_keys[i], sorted_p_keys[i+1]
        if points_for_interp[p1] < points_for_interp[p0]:
            logger.warning(f"Correcting non-monotonicity at percentile {p1*100:.1f}% after bound clamping.")
            points_for_interp[p1] = points_for_interp[p0]


    # --- Interpolate ---
    # Create the x-axis for the 201 points (values we want to find percentiles for)
    # This requires careful thought: Do we use linear scale or log scale based on zero_point?
    # The template used a potentially log-scaled generation. Let's use simple linear for now.
    # A more robust solution might inspect 'deriv_ratio_value_type' if available in question_details.
    cdf_xaxis_values = np.linspace(range_min_eff, range_max_eff, 201).tolist()

    # Invert the points: value -> percentile (needed for interpolation)
    value_to_percentile_map = {v: p for p, v in sorted(points_for_interp.items())} # Sort by percentile first

    # Interpolate percentiles for each value on our x-axis
    # Using numpy's interpolation is easiest
    known_values = sorted(value_to_percentile_map.keys())
    known_percentiles = [value_to_percentile_map[v] for v in known_values]

    # Ensure known_values are unique for interpolation
    unique_known_values = []
    unique_known_percentiles = []
    last_val = None
    for val, perc in zip(known_values, known_percentiles):
         if val != last_val:
              unique_known_values.append(val)
              unique_known_percentiles.append(perc)
              last_val = val
         else: # Average percentiles for duplicate values if they occur
             if unique_known_percentiles:
                 unique_known_percentiles[-1] = (unique_known_percentiles[-1] + perc) / 2.0


    if len(unique_known_values) < 2:
        logger.error(f"Cannot interpolate CDF with < 2 unique known values. Values: {unique_known_values}")
        return [0.5] * 201 # Fallback

    continuous_cdf = np.interp(cdf_xaxis_values, unique_known_values, unique_known_percentiles).tolist()

    # Final checks on CDF: should be ~0 at start, ~1 at end, and non-decreasing
    if not (0.0 <= continuous_cdf[0] <= 0.1): logger.warning(f"CDF start value unusual: {continuous_cdf[0]:.3f}")
    if not (0.9 <= continuous_cdf[-1] <= 1.0): logger.warning(f"CDF end value unusual: {continuous_cdf[-1]:.3f}")
    for i in range(len(continuous_cdf) - 1):
         if continuous_cdf[i+1] < continuous_cdf[i] - 1e-6: # Allow small float tolerance
              logger.warning(f"CDF non-monotonic at index {i+1}: {continuous_cdf[i+1]:.4f} < {continuous_cdf[i]:.4f}. Fixing.")
              continuous_cdf[i+1] = continuous_cdf[i] # Simple fix: flatten

    return continuous_cdf


############### QUESTION TYPE SPECIFIC FUNCTIONS ###############

async def get_binary_gemini_prediction(question_details: dict) -> tuple[Optional[float], str]:
    """Generates forecast and rationale for a binary question using Gemini."""

    prompt = clean_indents(f"""
        **Objective:** Analyze the provided binary forecasting question and generate a probabilistic forecast as a JSON object containing the probability for 'Yes' and detailed reasoning. Use web search results to inform your analysis.

        **Question Details:**
        * **Title:** {question_details['title']}
        * **URL:** {question_details['page_url']}
        * **Resolution Criteria:** {question_details['resolution_criteria']}
        * **Background:** {question_details['description']}
        * **Publish Time:** {question_details['publish_time']}
        * **Close Time:** {question_details['close_time']}
        * **Resolve Time:** {question_details['resolve_time']}
        * **Fine Print:** {question_details['fine_print']}
        * **Today's Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

        **Task:**
        1.  **Analyze:** Carefully consider the question, its resolution criteria, and relevant context (timeframes, definitions).
        2.  **Research (Internal Tool):** Use your search tool capabilities to find relevant historical data, base rates, expert opinions, or analogous situations. Summarize key findings relevant to estimating the probability IN THE 'HistoricalData' FIELD.
        3.  **Reason:** Develop a detailed rationale IN THE 'Rationale' FIELD explaining your thought process. Discuss key factors favoring 'Yes' or 'No', potential scenarios, uncertainties, and how search findings support your estimate.
        4.  **Forecast:** Estimate the probability (from 0.0 to 1.0) that the question will resolve to 'Yes' IN THE 'Prediction' FIELD.
        5.  **Format:** Return *only* a valid JSON object matching the specified schema. Ensure the JSON is the only text in your response.

        **JSON Schema:**
        ```json
        {{
          "Prediction": <number (0.0 to 1.0)>,
          "Rationale": "<string: Detailed reasoning>",
          "HistoricalData": "<string: Summary of key data/base rates found>"
        }}
        ```
        """)

    response_data = await call_gemini(prompt, binary_prediction_schema)

    if response_data:
        try:
            prediction_value = response_data.get('Prediction')
            rationale = response_data.get('Rationale', "Rationale missing.")
            historical_data = response_data.get('HistoricalData', "Historical data missing.")
            full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"

            if prediction_value is None or not isinstance(prediction_value, (float, int)):
                raise ValueError(f"Invalid 'Prediction' type: {type(prediction_value)}")

            prediction_float = max(0.001, min(0.999, float(prediction_value))) # Clamp slightly away from 0/1
            logger.info(f"Gemini Binary Forecast for Q {question_details['id']}: {prediction_float:.2%}")
            return prediction_float, full_reasoning

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error processing Gemini binary response for Q {question_details['id']}: {e}. Response: {response_data}", exc_info=True)
            # Fall through to failure case

    # --- Failure Case ---
    logger.error(f"Gemini forecast generation failed for binary Q {question_details['id']}.")
    return None, "Forecast generation failed." # Return None for prediction on failure


async def get_multiple_choice_gemini_prediction(question_details: dict) -> tuple[Optional[Dict[str, float]], str]:
    """Generates forecast and rationale for a multiple choice question using Gemini."""

    options_str = "\n".join([f"* {opt}" for opt in question_details['options']])
    prompt = clean_indents(f"""
        **Objective:** Analyze the provided multiple-choice forecasting question and generate a probabilistic forecast as a JSON object containing probabilities for each option and detailed reasoning. Use web search results.

        **Question Details:**
        * **Title:** {question_details['title']}
        * **URL:** {question_details['page_url']}
        * **Resolution Criteria:** {question_details['resolution_criteria']}
        * **Background:** {question_details['description']}
        * **Available Options:**
{options_str}
        * **Publish Time:** {question_details['publish_time']}
        * **Close Time:** {question_details['close_time']}
        * **Resolve Time:** {question_details['resolve_time']}
        * **Fine Print:** {question_details['fine_print']}
        * **Today's Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

        **Task:**
        1.  **Analyze:** Consider the question, criteria, and each option.
        2.  **Research (Internal Tool):** Use search to find info supporting/refuting each option. Summarize findings in 'HistoricalData'.
        3.  **Reason:** Develop a detailed rationale in 'Rationale' for the probability distribution. Discuss factors, scenarios, uncertainties, research findings for each option.
        4.  **Forecast:** Assign a probability (0.0 to 1.0) to *each* option IN THE 'Prediction' FIELD. The sum *must* equal 1.0.
        5.  **Format:** Return *only* a valid JSON object matching the schema below. Include probabilities for *all* options listed above.

        **JSON Schema:**
        ```json
        {{
          "Prediction": [
            {{"option": "<string: option_1>", "probability": <number (0.0 to 1.0)>}},
            {{"option": "<string: option_2>", "probability": <number (0.0 to 1.0)>}},
            ...
          ],
          "Rationale": "<string: Detailed reasoning>",
          "HistoricalData": "<string: Summary of key data/trends>"
        }}
        ```
        """)

    response_data = await call_gemini(prompt, multiple_choice_prediction_schema)

    if response_data:
        try:
            prediction_list = response_data.get('Prediction')
            rationale = response_data.get('Rationale', "Rationale missing.")
            historical_data = response_data.get('HistoricalData', "Historical data missing.")
            full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"

            if not isinstance(prediction_list, list):
                raise ValueError(f"Invalid 'Prediction' format: expected list, got {type(prediction_list)}")

            # Convert list of dicts to dict {option: probability} and validate/normalize
            formatted_prediction: Dict[str, float] = {}
            received_options_set = set()
            total_prob = 0.0
            original_options_set = set(question_details['options'])

            for item in prediction_list:
                if not isinstance(item, dict) or 'option' not in item or 'probability' not in item:
                    logger.warning(f"Invalid item format in MC prediction list: {item}. Skipping.")
                    continue
                option_name = item['option']
                prob_value = item['probability']

                if option_name not in original_options_set:
                    logger.warning(f"Received probability for unknown option '{option_name}' in Q {question_details['id']}. Ignoring.")
                    continue
                if option_name in received_options_set:
                    logger.warning(f"Duplicate probability for option '{option_name}' in Q {question_details['id']}. Using first.")
                    continue
                if not isinstance(prob_value, (float, int)):
                    raise ValueError(f"Invalid probability type for option '{option_name}': {type(prob_value)}")

                prob_float = max(0.0, min(1.0, float(prob_value))) # Clamp individual probs
                formatted_prediction[option_name] = prob_float
                received_options_set.add(option_name)
                total_prob += prob_float

            # Check for missing options
            missing_options = original_options_set - received_options_set
            if missing_options:
                logger.warning(f"Gemini MC response missing options: {missing_options}. Assigning zero and normalizing.")
                for opt in missing_options:
                    formatted_prediction[opt] = 0.0
                # Recalculate total_prob if zeros were added (though it shouldn't change)
                total_prob = sum(formatted_prediction.values())

            # Normalize probabilities if sum is off (and > 0)
            if abs(total_prob - 1.0) > 1e-3: # Use tolerance
                 if total_prob > 1e-6: # Avoid division by zero
                     logger.warning(f"MC probabilities sum to {total_prob:.4f} for Q {question_details['id']}. Normalizing.")
                     for opt in formatted_prediction:
                         formatted_prediction[opt] /= total_prob
                     # Ensure sum is exactly 1 after normalization due to potential float issues
                     final_sum = sum(formatted_prediction.values())
                     if abs(final_sum - 1.0) > 1e-6:
                         formatted_prediction[list(formatted_prediction.keys())[-1]] += (1.0 - final_sum)

                 else: # Sum is zero or near-zero
                    logger.warning(f"MC probabilities sum to zero for Q {question_details['id']}. Assigning even distribution.")
                    num_options = len(question_details['options'])
                    if num_options > 0:
                        even_prob = 1.0 / num_options
                        formatted_prediction = {opt: even_prob for opt in question_details['options']}
                    else: # Should not happen
                        logger.error(f"Q {question_details['id']} has no options for even distribution fallback.")
                        formatted_prediction = {} # Cannot form payload

            logger.info(f"Gemini MC Forecast for Q {question_details['id']}: {formatted_prediction}")
            return formatted_prediction, full_reasoning

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error processing Gemini MC response for Q {question_details['id']}: {e}. Response: {response_data}", exc_info=True)
            # Fall through to failure case

    # --- Failure Case ---
    logger.error(f"Gemini forecast generation failed for MC Q {question_details['id']}.")
    num_options = len(question_details['options'])
    fallback_pred: Dict[str, float] = {}
    if num_options > 0:
        even_prob = 1.0 / num_options
        fallback_pred = {opt: even_prob for opt in question_details['options']}
        logger.warning(f"Using fallback even distribution for Q {question_details['id']}: {fallback_pred}")
    else:
        logger.error(f"Q {question_details['id']} has no options for fallback.")

    return fallback_pred if fallback_pred else None, "Forecast generation failed (even distribution used)."


async def get_numeric_gemini_prediction(question_details: dict) -> tuple[Optional[List[float]], str]:
    """Generates forecast (CDF) and rationale for a numeric question using Gemini."""

    # Extract bounds and scaling info
    scaling = question_details.get("possibilities", {}).get("scale", {}) # Safely access nested dict
    lower_bound = scaling.get("min")
    upper_bound = scaling.get("max")
    open_lower_bound = question_details.get("possibilities", {}).get("lower_bound_open", lower_bound is None)
    open_upper_bound = question_details.get("possibilities", {}).get("upper_bound_open", upper_bound is None)
    zero_point = scaling.get("zero_point") # Might be None
    unit_of_measure = question_details.get("possibilities", {}).get("deriv_ratio_value_type", "") or "units"


    # Create bound messages
    lb_msg = "There is no lower bound." if open_lower_bound else f"IMPORTANT: Outcome cannot be lower than {lower_bound}."
    ub_msg = "There is no upper bound." if open_upper_bound else f"IMPORTANT: Outcome cannot be higher than {upper_bound}."


    prompt = clean_indents(f"""
        **Objective:** Analyze the provided numeric forecasting question and generate a probabilistic forecast as a JSON object containing specific percentiles (p10, p20, p40, p60, p80, p90) and detailed reasoning. Use web search results.

        **Question Details:**
        * **Title:** {question_details['title']}
        * **URL:** {question_details['page_url']}
        * **Resolution Criteria:** {question_details['resolution_criteria']}
        * **Background:** {question_details['description']}
        * **Publish Time:** {question_details['publish_time']}
        * **Close Time:** {question_details['close_time']}
        * **Resolve Time:** {question_details['resolve_time']}
        * **Fine Print:** {question_details['fine_print']}
        * {lb_msg}
        * {ub_msg}
        * **Units:** {unit_of_measure}
        * **Today's Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

        **Task:**
        1.  **Analyze:** Consider the question, bounds, criteria, context (units!).
        2.  **Research (Internal Tool):** Use search to find relevant historical data, trends, expert opinions, base rates. Summarize findings in 'HistoricalData'.
        3.  **Reason:** Develop a detailed rationale in 'Rationale'. Discuss factors, scenarios (low/median/high), uncertainties, and how research supports your distribution. Explain the distribution shape (skew, width).
        4.  **Forecast:** Estimate values for the 10th, 20th, 40th, 60th, 80th, and 90th percentiles IN THE 'Prediction' FIELD. Ensure percentiles are monotonically increasing (p10 <= p20 <= ... <= p90) and respect the question's bounds ({lb_msg}, {ub_msg}). Provide values as numbers, not strings with units.
        5.  **Format:** Return *only* a valid JSON object matching the schema below. Verify percentile values are numbers and strictly non-decreasing.

        **JSON Schema:**
        ```json
        {{
          "Prediction": {{
            "p10": <number>, "p20": <number>, "p40": <number>,
            "p60": <number>, "p80": <number>, "p90": <number>
          }},
          "Rationale": "<string: Detailed reasoning>",
          "HistoricalData": "<string: Summary of key data/trends>"
        }}
        ```
        """)

    response_data = await call_gemini(prompt, numeric_prediction_schema)

    if response_data:
        try:
            prediction_dict = response_data.get('Prediction')
            rationale = response_data.get('Rationale', "Rationale missing.")
            historical_data = response_data.get('HistoricalData', "Historical data missing.")
            full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"

            if not isinstance(prediction_dict, dict):
                raise ValueError(f"Invalid 'Prediction' format: expected dict, got {type(prediction_dict)}")

            required_keys = {'p10', 'p20', 'p40', 'p60', 'p80', 'p90'}
            if not required_keys.issubset(prediction_dict.keys()):
                missing = required_keys - set(prediction_dict.keys())
                raise ValueError(f"Missing required percentiles: {missing}")

            # Validate and convert percentile values to float {int_percentile: float_value}
            validated_percentiles = {}
            for key in sorted(required_keys, key=lambda x: int(x[1:])): # p10, p20...
                value = prediction_dict[key]
                try:
                    percentile_int = int(key[1:])
                    validated_percentiles[percentile_int] = float(value)
                except (ValueError, TypeError) as conv_err:
                     raise ValueError(f"Invalid number format for {key}: '{value}'. Error: {conv_err}")


            # Check monotonicity
            p_values = [validated_percentiles[p] for p in sorted(validated_percentiles.keys())]
            for i in range(len(p_values) - 1):
                # Use tolerance for float comparison
                if p_values[i] > p_values[i+1] + 1e-9:
                    raise ValueError(f"Percentiles not monotonically increasing: {p_values}")


            # Check bounds (clamp if necessary, though ideally model respects prompt)
            # Convert bounds to float for comparison, handle None
            lb_float = float(lower_bound) if lower_bound is not None else None
            ub_float = float(upper_bound) if upper_bound is not None else None

            if not open_lower_bound and lb_float is not None and validated_percentiles[10] < lb_float:
                logger.warning(f"Q {question_details['id']} p10 {validated_percentiles[10]} < LB {lb_float}. Clamping.")
                validated_percentiles[10] = lb_float
            if not open_upper_bound and ub_float is not None and validated_percentiles[90] > ub_float:
                logger.warning(f"Q {question_details['id']} p90 {validated_percentiles[90]} > UB {ub_float}. Clamping.")
                validated_percentiles[90] = ub_float

             # Re-check monotonicity after potential clamping
            p_values_clamped = [validated_percentiles[p] for p in sorted(validated_percentiles.keys())]
            for i in range(len(p_values_clamped) - 1):
                if p_values_clamped[i] > p_values_clamped[i+1] + 1e-9:
                     logger.warning(f"Q {question_details['id']} non-monotonic after clamping ({p_values_clamped[i]:.4f} > {p_values_clamped[i+1]:.4f}). Fixing.")
                     # Simple fix: set subsequent value equal to previous
                     for j in range(i + 1, len(p_values_clamped)):
                          validated_percentiles[sorted(validated_percentiles.keys())[j]] = max(
                               validated_percentiles[sorted(validated_percentiles.keys())[j]],
                               validated_percentiles[sorted(validated_percentiles.keys())[i]]
                          )


            # Generate the 201-point CDF for the payload
            cdf = generate_continuous_cdf(
                validated_percentiles,
                open_upper_bound, open_lower_bound,
                ub_float, lb_float, # Pass float versions
                float(zero_point) if zero_point is not None else None
            )

            logger.info(f"Gemini Numeric Forecast (Percentiles) for Q {question_details['id']}: {validated_percentiles}")
            # logger.debug(f"Generated CDF (first 10): {cdf[:10]}")
            return cdf, full_reasoning

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error processing Gemini numeric response for Q {question_details['id']}: {e}. Response: {response_data}", exc_info=True)
            # Fall through to failure case

    # --- Failure Case ---
    logger.error(f"Gemini forecast generation failed for numeric Q {question_details['id']}. Creating dummy CDF.")
    # Create dummy distribution similar to original main.py
    lb_float = float(lower_bound) if not open_lower_bound and lower_bound is not None else 0.0
    default_range = 10.0 # Simple default range
    ub_float = float(upper_bound) if not open_upper_bound and upper_bound is not None else (lb_float + default_range)

    if ub_float <= lb_float: ub_float = lb_float + 1.0 # Ensure range > 0

    dummy_percentiles = {
         p: lb_float + (ub_float - lb_float) * (p / 100.0)
         for p in [10, 20, 40, 60, 80, 90]
    }
    # Clamp dummy percentiles just in case
    if not open_lower_bound and lower_bound is not None: dummy_percentiles[10] = max(float(lower_bound), dummy_percentiles[10])
    if not open_upper_bound and upper_bound is not None: dummy_percentiles[90] = min(float(upper_bound), dummy_percentiles[90])
    # Ensure basic monotonicity
    keys = sorted(dummy_percentiles.keys())
    for i in range(len(keys) - 1):
         if dummy_percentiles[keys[i+1]] < dummy_percentiles[keys[i]]:
              dummy_percentiles[keys[i+1]] = dummy_percentiles[keys[i]]

    fallback_cdf = generate_continuous_cdf(
         dummy_percentiles, open_upper_bound, open_lower_bound,
         float(upper_bound) if upper_bound is not None else None,
         float(lower_bound) if lower_bound is not None else None,
         float(zero_point) if zero_point is not None else None
    )
    logger.warning(f"Using dummy CDF for Q {question_details['id']}")
    return fallback_cdf, "Forecast generation failed (dummy distribution used)."


################### FORECASTING ORCHESTRATION ###################

async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    skip_previously_forecasted_questions: bool,
) -> str:
    """Handles fetching, forecasting, and submitting for a single question."""
    post_details = get_post_details(post_id)
    if not post_details or "question" not in post_details:
        return f"-----------------------------------------------\nPost {post_id} Q {question_id}:\nFailed to fetch post details.\n"

    question_details = post_details["question"]
    # Add page_url if missing (construct it)
    if 'page_url' not in question_details:
         question_details['page_url'] = f"https://www.metaculus.com/questions/{question_id}/" # Or use post_id if more reliable

    title = question_details.get("title", "Unknown Title")
    question_type = question_details.get("type")

    summary_of_forecast = f"-----------------------------------------------\nQ {question_id} (Post {post_id}): {title}\nURL: {question_details['page_url']}\nType: {question_type}\n"

    if not question_type:
        summary_of_forecast += "Skipped: Unknown question type.\n"
        logger.error(f"Skipping Q {question_id}: Unknown type.")
        return summary_of_forecast

    if question_type == "multiple_choice":
        options = question_details.get("options", [])
        summary_of_forecast += f"Options: {[opt[:30] for opt in options]}\n" # Log truncated options

    if skip_previously_forecasted_questions and forecast_is_already_made(post_details):
        summary_of_forecast += "Skipped: Forecast already made by user.\n"
        logger.info(f"Skipping Q {question_id}: Already forecasted.")
        return summary_of_forecast

    forecast_result: Any = None
    comment = "Forecast failed."

    try:
        if question_type == "binary":
            forecast_result, comment = await get_binary_gemini_prediction(question_details)
        elif question_type == "numeric":
            forecast_result, comment = await get_numeric_gemini_prediction(question_details)
        elif question_type == "multiple_choice":
            forecast_result, comment = await get_multiple_choice_gemini_prediction(question_details)
        else:
            summary_of_forecast += f"Skipped: Unsupported question type '{question_type}'.\n"
            logger.warning(f"Skipping Q {question_id}: Unsupported type '{question_type}'.")
            return summary_of_forecast # Skip unsupported types

    except Exception as e:
         logger.error(f"Error during prediction generation for Q {question_id}: {e}", exc_info=True)
         comment = f"Error during prediction generation: {e}"
         forecast_result = None # Ensure result is None on error

    # Format summary based on forecast type
    if forecast_result is not None:
         if question_type == "numeric":
              summary_of_forecast += f"Forecast (CDF Start/Mid/End): {forecast_result[0]:.3f} / {forecast_result[100]:.3f} / {forecast_result[-1]:.3f}\n"
         elif question_type == "multiple_choice":
              summary_of_forecast += f"Forecast (MC): { {k: f'{v:.3f}' for k, v in forecast_result.items()} }\n"
         else: # Binary
              summary_of_forecast += f"Forecast (Binary): {forecast_result:.3f}\n"
    else:
         summary_of_forecast += "Forecast: FAILED\n"


    summary_of_forecast += f"Comment (Preview):\n```\n{comment[:300]}...\n```\n"
    save_report(post_id, question_id, f"# Forecast Report Q{question_id}\n\n## Details\n```json\n{json.dumps(question_details, indent=2, default=str)}\n```\n\n## Rationale\n{comment}")


    if submit_prediction and forecast_result is not None:
        try:
            forecast_payload = create_forecast_payload(forecast_result, question_type)
            if forecast_payload: # Only post if payload creation succeeded
                post_question_prediction(question_id, forecast_payload)
                # Post comment *after* prediction (or make configurable)
                # Limit comment length for API
                max_comment_len = 10000
                truncated_comment = comment if len(comment) <= max_comment_len else comment[:max_comment_len-3] + "..."
                post_question_comment(post_id, truncated_comment)
                summary_of_forecast += "Posted: Forecast and comment submitted to Metaculus.\n"
            else:
                 summary_of_forecast += "Posting Failed: Could not create valid forecast payload.\n"
                 logger.error(f"Did not post for Q {question_id} due to payload creation failure.")

        except Exception as post_err:
             logger.error(f"Error submitting forecast/comment for Q {question_id}: {post_err}")
             summary_of_forecast += f"Posting Failed: Error during submission ({post_err}).\n"
    elif not submit_prediction:
         summary_of_forecast += "Posting Skipped: submit_prediction is False.\n"
    elif forecast_result is None:
         summary_of_forecast += "Posting Skipped: Forecast generation failed.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    skip_previously_forecasted_questions: bool,
    submit_prediction: bool = True,
) -> None:
    """Asynchronously forecasts a list of questions."""
    tasks = [
        forecast_individual_question(
            q_id, post_id, submit_prediction, skip_previously_forecasted_questions
        )
        for q_id, post_id in open_question_id_post_id
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("\n" + "#" * 100 + "\nForecast Run Summary\n" + "#" * 100)
    errors = []
    for result, (q_id, post_id) in zip(results, open_question_id_post_id):
        if isinstance(result, Exception):
            error_msg = f"-----------------------------------------------\nPost {post_id} Q {q_id}:\nError: {result.__class__.__name__} {result}\nURL: https://www.metaculus.com/questions/{q_id}/\n" # Use q_id for question URL
            logger.error(f"Unhandled exception for Q {q_id}: {result}", exc_info=result)
            print(error_msg) # Also print error summary to console
            errors.append(error_msg)
        else:
            print(result) # Print the summary string returned by forecast_individual_question

    if errors:
        print("-----------------------------------------------\nErrors Encountered:\n")
        for err in errors:
            print(err)
        # Optionally raise a final exception to signal failure
        # raise RuntimeError(f"{len(errors)} errors occurred during forecasting.")
    logger.info("Finished forecasting run.")


######################### FINAL RUN (Simplified __main__) #########################
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run the Gemini forecasting bot (No Framework).")

    # --- Simplified Run Mode ---
    parser.add_argument(
        "--mode", type=str, choices=["tournament", "test_questions"], # Simplified choices
        default="tournament", help="Specify the run mode (tournament questions or fixed test questions)"
    )
    args = parser.parse_args()

    # --- Initial Checks ---
    if not METACULUS_TOKEN:
        logger.critical("METACULUS_TOKEN environment variable not set. Cannot interact with Metaculus.")
        exit(1)
    if not GEMINI_API_KEY:
        logger.critical("GEMINI_API_KEY environment variable not set. Cannot use Gemini.")
        exit(1)

    # --- Determine Question List and Skip Logic based on Mode ---
    question_list: list[tuple[int, int]] = []
    run_mode = args.mode
    effective_skip_previous: bool

    if run_mode == "test_questions":
        logger.info(f"Running in mode: {run_mode} on {len(EXAMPLE_QUESTIONS)} example questions.")
        question_list = EXAMPLE_QUESTIONS
        effective_skip_previous = False # ALWAYS run on example questions
        logger.info("Skip previously forecasted: False (fixed for test_questions mode)")
    else: # mode == "tournament"
        target_tournament_id = Q1_2025_AI_BENCHMARKING_ID # Use default tournament ID
        logger.info(f"Running in mode: {run_mode}. Fetching open questions from tournament ID: {target_tournament_id}")
        question_list = get_open_question_ids_from_tournament(target_tournament_id)
        effective_skip_previous = True # Default skip for tournament mode
        logger.info("Skip previously forecasted: True (default for tournament mode)")


    if not question_list:
         logger.warning(f"No questions found for mode '{run_mode}'. Exiting.")
         exit(0)

    # --- Log Run Configuration ---
    logger.info(f"Starting forecast run for {len(question_list)} questions.")
    logger.info(f"Run Mode: {args.mode}")
    # Removed logging for args.skip_previous, logged effective value above

    # --- Execute Forecasting ---
    try:
        asyncio.run(
            forecast_questions(
                question_list,
                skip_previously_forecasted_questions=effective_skip_previous, # Use determined value
            )
        )
    except Exception as main_err:
         logger.critical(f"An unexpected error occurred in the main execution loop: {main_err}", exc_info=True)
    finally:
         logger.info("--- End of Script ---")