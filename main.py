import argparse
import asyncio
import logging
import os
import json
from datetime import datetime
from typing import Literal, Optional, Dict, Any, Tuple, List

# --- Google Generative AI Imports ---
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

# --- Forecasting Tools Imports ---
from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm, # For fallback
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor, # For fallback parsing
    ReasonedPrediction,
    clean_indents,
)
# Ensure LiteLLM/OpenAI dependencies are available if GeneralLlm uses them
try:
    import litellm
except ImportError:
    print("Warning: LiteLLM not installed. Fallback LLM functionality might be limited.")

# --- Logger Setup ---
logger = logging.getLogger(__name__)

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


class GeminiForecaster(ForecastBot):
    """
    A forecasting bot using the Google AI Gemini API (API Key based)
    with Google Search tool, and an optional fallback LLM.
    Configuration prioritizes environment variables.
    """
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    # --- Default Gemini Configuration ---
    DEFAULT_GEMINI_MODEL = "gemini-2.5-pro-preview-03-25" # Changed default for API Key model

    def __init__(
        self,
        # --- Bot Behavior Config ---
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = True,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = True,
        # --- Optional Gemini Model Override ---
        gemini_model_name_override: Optional[str] = None,
        # --- Fallback LLM configuration ---
        fallback_llm_model: Optional[str] = None,
        fallback_llm_temp: float = 0.3,
        fallback_llm_timeout: int = 60,
    ):
        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
        )

        # --- Configure Gemini Client (API Key: Env Var) ---
        self.api_key = os.getenv("GEMINI_API_KEY")
        
        self.gemini_client = None # Initialize client as None

        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini calls will fail.")
        else:
            try:
                # Initialize client using genai.Client(api_key=...)
                self.gemini_client = genai.Client(api_key=self.api_key)
                logger.info("Gemini client initialized using API Key.")
            except Exception as e:
                 logger.error(f"Failed to initialize Gemini client with API key: {e}", exc_info=True)
                 self.gemini_client = None # Ensure client is None on failure

        # --- Determine Gemini Model Name (Env Var > Argument > Default) ---
        self.model_name = os.getenv("GEMINI_MODEL_NAME")
        if gemini_model_name_override:
            self.model_name = gemini_model_name_override
            logger.info(f"Using Gemini Model Name from argument override: {self.model_name}")
        elif self.model_name:
             logger.info(f"Using Gemini Model Name from environment variable: {self.model_name}")
        else:
            self.model_name = self.DEFAULT_GEMINI_MODEL
            logger.info(f"Using default Gemini Model Name: {self.model_name}")

        # --- Fallback LLM Initialization (Unchanged) ---
        self.fallback_llm = None
        # ...(fallback init logic remains the same)...
        if fallback_llm_model:
            try:
                self.fallback_llm = GeneralLlm(...)
                logger.info(f"Fallback LLM initialized: {fallback_llm_model}")
            except Exception as e: logger.error(f"Failed to initialize fallback LLM: {e}")
        else: logger.info("No fallback LLM configured.")


    async def run_research(self, question: MetaculusQuestion) -> str:
            """
            Placeholder implementation required by the ForecastBot abstract base class.
            This bot relies on Gemini's internal search tool or skips dedicated research.
            """
            logger.info(f"Skipping external research step for {question.page_url} (handled by Gemini tool or omitted).")
            # Return an empty string, matching the expected return type hint.
            return ""
    

    async def _call_gemini(self, prompt: str, response_schema: types.Schema) -> Optional[Dict[str, Any]]:
        """
        Calls the Gemini API using the genai.Client instance (API Key).
        Attempts call via client.models.generate_content(...).
        Returns parsed JSON dict on success, None on failure.
        """
        if not self.gemini_client:
            logger.error("Gemini client not initialized. Cannot make API call.")
            return None

        async with self._concurrency_limiter:
            try:
                # Use the model name determined in __init__
                model_id_for_call = f"models/{self.model_name}" # Format needed for direct API calls
                logger.debug(f"Sending request to Gemini model '{model_id_for_call}' via genai.Client...")

                # Construct the request parts
                contents = [types.Content(role="user", parts=[prompt])]

                # Define tools (Search)
                tools = [types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]

                # Construct the generation config
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=16300,
                    response_mime_type="application/json", # MUST request JSON
                    response_schema=response_schema, # Pass the specific schema
                )
                # Make the API call using client.models.generate_content
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content, # Calling method on client.models
                    model=model_id_for_call, # Pass model name here
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
# --- _run_forecast_on_binary (No Fallback) ---
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion , research:str 
    ) -> ReasonedPrediction[float]:
        """
        Generates a forecast for a binary question using Gemini (API Key).
        No fallback LLM is used. Returns 0.5 on failure.
        """
        # --- Construct Gemini Prompt ---
        gemini_prompt = clean_indents(f"""
            **Objective:** Analyze the provided binary forecasting question and generate a probabilistic forecast as a JSON object containing the probability for 'Yes' and detailed reasoning. Use web search results to inform your analysis.

            **Question Details:**
            * **Title:** {question.title}
            * **URL:** {question.page_url}
            * **Resolution Criteria:** {question.resolution_criteria}
            * **Background:** {question.background}
            * **Publish Time:** {question.publish_time}
            * **Close Time:** {question.close_time}
            * **Resolve Time:** {question.resolve_time}

            **Task:**
            1.  **Analyze:** Carefully consider the question, its resolution criteria, and relevant context (timeframes, definitions).
            2.  **Research (Internal Tool):** Use your search tool capabilities to find relevant historical data, base rates, expert opinions, or analogous situations. Summarize key findings relevant to estimating the probability.
            3.  **Reason:** Develop a detailed rationale explaining your thought process. Discuss the key factors favoring 'Yes' or 'No', potential scenarios, uncertainties, and how the research findings support your probability estimate.
            4.  **Forecast:** Estimate the probability (from 0.0 to 1.0) that the question will resolve to 'Yes'.
            5.  **Format:** Return *only* a valid JSON object matching the specified schema:
                ```json
                {{
                  "Prediction": <number (0.0 to 1.0)>,
                  "Rationale": "<string: Detailed reasoning>",
                  "HistoricalData": "<string: Summary of key data/base rates found>"
                }}
                ```
            Ensure the JSON is the only text in your response.
            """)

        response_data = await self._call_gemini(gemini_prompt, binary_prediction_schema)

        if response_data:
            try:
                prediction_value = response_data.get('Prediction')
                rationale = response_data.get('Rationale', "Rationale not provided in JSON.")
                historical_data = response_data.get('HistoricalData', "Historical data summary not provided.")
                full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"

                if prediction_value is None or not isinstance(prediction_value, (float, int)):
                    raise ValueError(f"Invalid 'Prediction' value type: {type(prediction_value)}")

                prediction_float = max(0.0, min(1.0, float(prediction_value))) # Clamp value

                logger.info(f"Gemini Forecasted URL {question.page_url} as {prediction_float:.2%}")
                return ReasonedPrediction(prediction_value=prediction_float, reasoning=full_reasoning)

            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error processing Gemini binary response for {question.page_url}: {e}. Response: {response_data}", exc_info=True)
                # Fall through to failure case if processing fails

        # --- Failure Case (No Fallback) ---
        logger.error(f"Gemini forecast generation failed for binary question {question.page_url}.")
        return ReasonedPrediction(prediction_value=0.5, reasoning="Forecast generation failed.")

    # --- _run_forecast_on_multiple_choice (No Fallback) ---
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion , research:str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """
        Generates a forecast for a multiple-choice question using Gemini (API Key).
        No fallback LLM is used. Returns an even distribution on failure.
        """
        # --- Construct Gemini Prompt ---
        options_str = "\n".join([f"* {opt}" for opt in question.options])
        gemini_prompt = clean_indents(f"""
            **Objective:** Analyze the provided multiple-choice forecasting question and generate a probabilistic forecast as a JSON object containing probabilities for each option and detailed reasoning. Use web search results to inform your analysis.

            **Question Details:**
            * **Title:** {question.title}
            * **URL:** {question.page_url}
            * **Resolution Criteria:** {question.resolution_criteria}
            * **Background:** {question.background}
            * **Available Options:**
            {options_str}
            * **Publish Time:** {question.publish_time}
            * **Close Time:** {question.close_time}
            * **Resolve Time:** {question.resolve_time}

            **Task:**
            1.  **Analyze:** Carefully consider the question, resolution criteria, and each available option.
            2.  **Research (Internal Tool):** Use your search tool capabilities to find information supporting or refuting each option. Look for relevant data, expert opinions, base rates, or analogous situations. Summarize key findings relevant to estimating the probabilities.
            3.  **Reason:** Develop a detailed rationale explaining your thought process for the probability distribution. Discuss the key factors influencing the likelihood of each option, potential scenarios, uncertainties, and how the research findings support your estimates. Address why certain options are more or less likely than others.
            4.  **Forecast:** Assign a probability (from 0.0 to 1.0) to *each* option provided. The sum of all assigned probabilities *must* equal 1.0.
            5.  **Format:** Return *only* a valid JSON object matching the specified schema:
                ```json
                {{
                  "Prediction": [
                    {{"option": "<string: option_1>", "probability": <number (0.0 to 1.0)>}},
                    {{"option": "<string: option_2>", "probability": <number (0.0 to 1.0)>}},
                    ...
                  ],
                  "Rationale": "<string: Detailed reasoning for the distribution>",
                  "HistoricalData": "<string: Summary of key data/trends found>"
                }}
                ```
            Ensure the JSON is the only text in your response and includes probabilities for *all* options listed above, summing to 1.0.
            """)

        response_data = await self._call_gemini(gemini_prompt, multiple_choice_prediction_schema)

        if response_data:
            try:
                prediction_list = response_data.get('Prediction')
                rationale = response_data.get('Rationale', "Rationale not provided in JSON.")
                historical_data = response_data.get('HistoricalData', "Historical data summary not provided.")
                full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"


                if not isinstance(prediction_list, list):
                    raise ValueError(f"Invalid 'Prediction' format: expected list, got {type(prediction_list)}")

                # Validate and normalize the received prediction list
                formatted_prediction: PredictedOptionList = []
                received_options = set()
                total_prob = 0.0

                for item in prediction_list:
                    if not isinstance(item, dict) or 'option' not in item or 'probability' not in item:
                        raise ValueError(f"Invalid item format in 'Prediction' list: {item}")
                    option_name = item['option']
                    prob_value = item['probability']

                    if option_name not in question.options:
                        logger.warning(f"Received probability for an unknown option '{option_name}' in {question.page_url}. Ignoring.")
                        continue
                    if option_name in received_options:
                        logger.warning(f"Received duplicate probability for option '{option_name}' in {question.page_url}. Using first instance.")
                        continue

                    if not isinstance(prob_value, (float, int)):
                         raise ValueError(f"Invalid probability type for option '{option_name}': {type(prob_value)}")

                    prob_float = max(0.0, min(1.0, float(prob_value))) # Clamp value
                    formatted_prediction.append((option_name, prob_float))
                    received_options.add(option_name)
                    total_prob += prob_float

                # Check if all options were covered
                missing_options = set(question.options) - received_options
                if missing_options:
                     logger.warning(f"Gemini response for {question.page_url} missing options: {missing_options}. Assigning zero probability.")
                     for opt in missing_options:
                         formatted_prediction.append((opt, 0.0))

                # Normalize probabilities if they don't sum exactly to 1.0 (within tolerance)
                if not (0.99 < total_prob < 1.01) and total_prob > 0:
                    logger.warning(f"Probabilities for {question.page_url} sum to {total_prob}. Normalizing.")
                    normalized_prediction: PredictedOptionList = []
                    for opt, prob in formatted_prediction:
                        normalized_prediction.append((opt, prob / total_prob))
                    formatted_prediction = normalized_prediction
                elif total_prob == 0 and len(question.options) > 0: # Handle case where all probs were 0
                     logger.warning(f"All probabilities for {question.page_url} were zero. Assigning even distribution.")
                     num_options = len(question.options)
                     even_prob = 1.0 / num_options
                     formatted_prediction = [(opt, even_prob) for opt in question.options]
                elif not (0.99 < total_prob < 1.01): # Handle sum outside tolerance but not zero
                     raise ValueError(f"Probabilities sum ({total_prob}) is invalid after processing for {question.page_url}.")


                logger.info(f"Gemini Forecasted URL {question.page_url} as {formatted_prediction}")
                return ReasonedPrediction(prediction_value=formatted_prediction, reasoning=full_reasoning)

            except (ValueError, TypeError, KeyError) as e:
                 logger.error(f"Error processing Gemini multiple choice response for {question.page_url}: {e}. Response: {response_data}", exc_info=True)
                 # Fall through to failure case if processing fails

        # --- Failure Case (No Fallback) ---
        logger.error(f"Gemini forecast generation failed for multiple choice question {question.page_url}.")
        num_options = len(question.options)
        fallback_pred: PredictedOptionList = []
        if num_options > 0:
            even_prob = 1.0 / num_options
            fallback_pred = [(opt, even_prob) for opt in question.options]
        else: # Should not happen with valid questions
             logger.error(f"Question {question.page_url} has no options defined.")

        return ReasonedPrediction(prediction_value=fallback_pred, reasoning="Forecast generation failed (even distribution used).")


    # --- _run_forecast_on_numeric (No Fallback) ---
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion , research:str
    ) -> ReasonedPrediction[NumericDistribution]:
        """
        Generates a forecast for a numeric question using Gemini (API Key).
        No fallback LLM is used. Returns a dummy distribution on failure.
        """
        # --- Construct Gemini Prompt ---
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        gemini_prompt = clean_indents(f"""
            **Objective:** Analyze the provided forecasting question and generate a probabilistic forecast as a JSON object containing specific percentiles (p10, p20, p40, p60, p80, p90) and detailed reasoning. Use web search results to inform your analysis.

            **Question Details:**
            * **Title:** {question.title}
            * **URL:** {question.page_url}
            * **Resolution Criteria:** {question.resolution_criteria}
            * **Background:** {question.background}
            * **Publish Time:** {question.publish_time}
            * **Close Time:** {question.close_time}
            * **Resolve Time:** {question.resolve_time}
            * {upper_bound_message}
            * {lower_bound_message}
            * **Possible Range:** {question.possibilities.get("scale", {}).get("min", "N/A")} to {question.possibilities.get("scale", {}).get("max", "N/A")} {question.possibilities.get("scale", {}).get("deriv_ratio_value_type", "")}

            **Task:**
            1.  **Analyze:** Carefully consider the question, its bounds, resolution criteria, and any relevant context (timeframes, definitions, units).
            2.  **Research (Internal Tool):** Use your search tool capabilities to find relevant historical data, trends, expert opinions, base rates, or analogous situations. Summarize key findings relevant to estimating the outcome value distribution.
            3.  **Reason:** Develop a detailed rationale explaining your thought process. Discuss the key factors influencing the likely outcome, potential scenarios (low, median, high), uncertainties, and how the research findings support your distribution. Address the specific percentiles. Explain the shape of your distribution (e.g., skewed left/right, wide/narrow uncertainty).
            4.  **Forecast:** Estimate the values corresponding to the 10th, 20th, 40th, 60th, 80th, and 90th percentiles of the probability distribution for the final outcome. Ensure the percentiles are monotonically increasing (p10 <= p20 <= ... <= p90) and respect the question's bounds ({lower_bound_message}, {upper_bound_message}).
            5.  **Format:** Return *only* a valid JSON object matching the specified schema:
                ```json
                {{
                  "Prediction": {{
                    "p10": <number>,
                    "p20": <number>,
                    "p40": <number>,
                    "p60": <number>,
                    "p80": <number>,
                    "p90": <number>
                  }},
                  "Rationale": "<string: Detailed reasoning>",
                  "HistoricalData": "<string: Summary of key data/trends found>"
                }}
                ```
            Ensure the JSON is the only text in your response. Verify percentile values are numbers and strictly non-decreasing.
            """)

        response_data = await self._call_gemini(gemini_prompt, numeric_prediction_schema)

        if response_data:
            try:
                prediction_dict = response_data.get('Prediction')
                rationale = response_data.get('Rationale', "Rationale not provided in JSON.")
                historical_data = response_data.get('HistoricalData', "Historical data summary not provided.")
                full_reasoning = f"Rationale:\n{rationale}\n\nHistorical Data/Analysis:\n{historical_data}"

                if not isinstance(prediction_dict, dict):
                    raise ValueError(f"Invalid 'Prediction' format received: expected dict, got {type(prediction_dict)}")

                required_percentiles_keys = {'p10', 'p20', 'p40', 'p60', 'p80', 'p90'}
                if not required_percentiles_keys.issubset(prediction_dict.keys()):
                    missing = required_percentiles_keys - set(prediction_dict.keys())
                    raise ValueError(f"Missing required percentiles in 'Prediction': {missing}")

                # Convert percentile keys (like 'p10') to integer keys (like 10) and ensure values are floats
                # Store temporarily to check monotonicity later
                raw_percentiles = {}
                for key in sorted(required_percentiles_keys, key=lambda x: int(x[1:])): # Process in order p10, p20...
                    value = prediction_dict[key]
                    try:
                        percentile_int = int(key[1:]) # Remove 'p' and convert to int
                        if isinstance(value, (int, float)):
                            raw_percentiles[percentile_int] = float(value)
                        elif isinstance(value, str):
                             try:
                                 raw_percentiles[percentile_int] = float(value)
                             except ValueError:
                                 raise ValueError(f"Percentile value for {key} is not a valid number: '{value}'")
                        else:
                             raise ValueError(f"Invalid type for percentile value {key}: {type(value)}")
                    except (ValueError, TypeError) as conv_err:
                        raise ValueError(f"Error processing percentile {key}={value}: {conv_err}")

                # Check monotonicity
                percentile_values = [raw_percentiles[p] for p in sorted(raw_percentiles.keys())]
                for i in range(len(percentile_values) - 1):
                     # Use a small tolerance for float comparisons if necessary, though strict <= is usually fine
                     if percentile_values[i] > percentile_values[i+1] + 1e-9: # Add tolerance
                          raise ValueError(f"Percentiles are not monotonically increasing: {percentile_values}")

                # Check bounds (if they exist)
                if not question.open_lower_bound and raw_percentiles[10] < question.lower_bound:
                      logger.warning(f"p10 value {raw_percentiles[10]} is below lower bound {question.lower_bound}. Clamping.")
                      raw_percentiles[10] = float(question.lower_bound)
                      # Potentially re-check monotonicity after clamping if needed
                if not question.open_upper_bound and raw_percentiles[90] > question.upper_bound:
                      logger.warning(f"p90 value {raw_percentiles[90]} is above upper bound {question.upper_bound}. Clamping.")
                      raw_percentiles[90] = float(question.upper_bound)
                      # Potentially re-check monotonicity after clamping if needed


                # Pass necessary bounds/info from the question object along with validated percentiles
                numeric_dist = NumericDistribution(
                    declared_percentiles=raw_percentiles, # Use the validated dict
                    open_upper_bound=question.open_upper_bound,
                    open_lower_bound=question.open_lower_bound,
                    upper_bound=question.upper_bound,
                    lower_bound=question.lower_bound,
                    zero_point=question.zero_point
                )

                logger.info(f"Gemini Forecasted URL {question.page_url} distribution: {numeric_dist.declared_percentiles}")
                return ReasonedPrediction(prediction_value=numeric_dist, reasoning=full_reasoning)

            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error processing Gemini numeric response for {question.page_url}: {e}. Response: {response_data}", exc_info=True)
                # Fall through to failure case if processing fails


        # --- Failure Case (No Fallback) ---
        logger.error(f"Gemini forecast generation failed for numeric question {question.page_url}. Creating dummy distribution.")

        # Calculate low/high bounds for the dummy distribution, ensuring they are floats
        low = float(question.lower_bound) if not question.open_lower_bound else 0.0
        # Define high relative to low if upper is open, choose a sensible default range maybe based on zero_point if available?
        default_range = 10.0 if question.zero_point is None else abs(question.zero_point * 2) if question.zero_point != 0 else 10.0
        high = float(question.upper_bound) if not question.open_upper_bound else (low + default_range)

        # Handle cases where bounds might be illogical or both open
        if not question.open_lower_bound and not question.open_upper_bound and high <= low:
            high = low + 1.0 # Ensure minimal range if bounds defined but invalid
        elif question.open_lower_bound and question.open_upper_bound:
             # Use a generic range if both open, maybe centered around zero_point?
             zp = float(question.zero_point) if question.zero_point is not None else 0.0
             low = zp - default_range / 2
             high = zp + default_range / 2


        # Create the dummy percentile dictionary (using integer keys) - simple linear spread
        dummy_percentiles = {
            10: low + (high - low) * 0.1,
            20: low + (high - low) * 0.2,
            40: low + (high - low) * 0.4,
            60: low + (high - low) * 0.6,
            80: low + (high - low) * 0.8,
            90: low + (high - low) * 0.9,
        }
        # Clamp to bounds just in case calculation goes outside
        if not question.open_lower_bound:
            dummy_percentiles[10] = max(float(question.lower_bound), dummy_percentiles[10])
        if not question.open_upper_bound:
             dummy_percentiles[90] = min(float(question.upper_bound), dummy_percentiles[90])
        # Ensure basic monotonicity for the dummy values after potential clamping/calculation issues
        keys = sorted(dummy_percentiles.keys())
        for i in range(len(keys) - 1):
             if dummy_percentiles[keys[i+1]] < dummy_percentiles[keys[i]]:
                  dummy_percentiles[keys[i+1]] = dummy_percentiles[keys[i]]


        # Create the distribution object, passing ALL required fields from the original question
        fallback_dist = NumericDistribution(
            declared_percentiles=dummy_percentiles, # Provide the basic linear spread
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,      # Pass original bounds
            lower_bound=question.lower_bound,      # Pass original bounds
            zero_point=question.zero_point        # Pass original zero_point
        )

        logger.warning(f"Using dummy distribution for {question.page_url}: {fallback_dist.declared_percentiles}")
        return ReasonedPrediction(prediction_value=fallback_dist, reasoning="Forecast generation failed (dummy distribution used).")


# --- Main Execution Block (Updated Arg Parsing) ---
if __name__ == "__main__":
    # --- Basic Logging Config ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Suppress Google API core logs unless needed for debugging
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run the GeminiForecaster (API Key) forecasting system with optional fallback"
    )
    # --- Run Mode ---
    parser.add_argument(
        "--mode", type=str, choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament", help="Specify the run mode"
    )
    # --- Gemini Config (Optional Model Override) ---
    # Project ID and Location are no longer needed for API Key model
    parser.add_argument("--model", type=str, default=None, help="Override GEMINI_MODEL_NAME env var")
    # --- Fallback Config ---
    parser.add_argument("--fallback-model", type=str, default=None, help="Fallback LLM model name. Set to empty/None to disable.")
    parser.add_argument("--fallback-temp", type=float, default=0.3, help="Temperature for fallback LLM")
    parser.add_argument("--fallback-timeout", type=int, default=600, help="Timeout for fallback LLM call")
    # --- General Bot Config ---
    parser.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True, help="Publish predictions to Metaculus")
    parser.add_argument("--skip-previous", action=argparse.BooleanOptionalAction, default=True, help="Skip questions already forecasted")
    parser.add_argument("--predictions-per-question", type=int, default=1, help="Number of Gemini prediction attempts per question")

    args = parser.parse_args()

    # Handle disabling fallback
    fallback_model_name = args.fallback_model
    if fallback_model_name and fallback_model_name.lower() in ["", "none", "null"]:
        fallback_model_name = None
        logger.info("Fallback LLM explicitly disabled via command line.")

    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = args.mode

    # --- Check for Gemini API Key ---
    if not os.getenv("GEMINI_API_KEY"):
        logger.critical("CRITICAL: GEMINI_API_KEY environment variable is not set. The bot cannot function.")
        # Optionally exit here if Gemini is absolutely required
        # exit(1) # Uncomment to exit if API key is missing

    # --- Initialize the Bot ---
    gemini_bot = GeminiForecaster(
        predictions_per_research_report=args.predictions_per_question,
        publish_reports_to_metaculus=args.publish,
        folder_to_save_reports_to="gemini_forecast_reports",
        skip_previously_forecasted_questions=args.skip_previous,
        # Pass potential model override (can be None)
        gemini_model_name_override=args.model,
        # Fallback config
        fallback_llm_model=fallback_model_name,
        fallback_llm_temp=args.fallback_temp,
        fallback_llm_timeout=args.fallback_timeout,
    )

    # --- Run Forecasting based on Mode ---
    forecast_reports = []
    # ...(Rest of the execution logic identical to v3)...
    try:
        if run_mode == "tournament":
            logger.info(f"Running in tournament mode for competition ID: {MetaculusApi.CURRENT_AI_COMPETITION_ID}")
            forecast_reports = asyncio.run(
                gemini_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
        elif run_mode == "quarterly_cup":
            logger.info(f"Running in quarterly cup mode for competition ID: {MetaculusApi.CURRENT_QUARTERLY_CUP_ID}")
            gemini_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                gemini_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
                )
            )
        elif run_mode == "test_questions":
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
            ]
            logger.info(f"Running in test mode on questions: {EXAMPLE_QUESTIONS}")
            gemini_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                gemini_bot.forecast_questions(questions, return_exceptions=True)
            )

    except Exception as main_err:
         logger.critical(f"An error occurred during the main execution: {main_err}", exc_info=True)
    finally:
        # --- Log Summary ---
        if forecast_reports:
            logger.info("--- Forecast Run Summary ---")
            GeminiForecaster.log_report_summary(forecast_reports) # type: ignore
            logger.info("--- End of Summary ---")
        else:
            logger.warning("No forecast reports were generated.")