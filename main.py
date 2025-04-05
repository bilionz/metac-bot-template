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
        if not self.api_key:
            logger.warning("GEMINI_API_KEY environment variable not set. Gemini calls will fail.")
        else:
            try:
                # Configure the API key globally for the genai library
                genai.configure(api_key=self.api_key)
                logger.info("Gemini API Key configured.")
            except Exception as e:
                 logger.error(f"Failed to configure Gemini API key: {e}")
                 self.api_key = None # Ensure key is None if config fails

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

        # --- Initialize Gemini Client ---
        self.gemini_client = None
        if self.api_key: # Only initialize if API key is configured
            try:
                # Initialize the GenerativeModel instance
                self.gemini_client = genai.GenerativeModel(
                    self.model_name,
                    # Default generation config - expecting JSON
                    generation_config=types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.95,
                        max_output_tokens=16000, # Adjust as needed
                        response_mime_type="application/json", # CRITICAL: Keep for structured output
                    ),
                    # Define tools (Google Search)
                    tools=[
                        types.Tool(google_search_retrieval=types.GoogleSearch()),
                    ]
                )
                logger.info(f"Gemini client initialized for model '{self.model_name}' using API Key.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client model '{self.model_name}': {e}")
                self.gemini_client = None
        else:
            logger.error("Cannot initialize Gemini client without an API Key.")

        # --- Fallback LLM Initialization (Unchanged) ---
        self.fallback_llm = None
        if fallback_llm_model:
            # Check for necessary API keys for fallback
            if "openai" in fallback_llm_model.lower() and not os.getenv("OPENAI_API_KEY"):
                 logger.warning(f"OPENAI_API_KEY not set. Fallback model {fallback_llm_model} may not work.")
            # Add checks for other keys (e.g., ANTHROPIC_API_KEY) if needed
            try:
                self.fallback_llm = GeneralLlm(
                    model=fallback_llm_model,
                    temperature=fallback_llm_temp,
                    timeout=fallback_llm_timeout,
                )
                logger.info(f"Fallback LLM initialized: {fallback_llm_model}")
            except Exception as e:
                logger.error(f"Failed to initialize fallback LLM ({fallback_llm_model}): {e}")
                self.fallback_llm = None
        else:
            logger.info("No fallback LLM configured.")


    async def _call_gemini(self, prompt: str, response_schema: types.Schema) -> Optional[Dict[str, Any]]:
        """
        Calls the Gemini API using the configured API key.
        Returns parsed JSON dict on success, None on failure.
        """
        if not self.gemini_client:
            logger.error("Gemini client not initialized. Cannot make API call.")
            return None

        async with self._concurrency_limiter:
            try:
                logger.debug(f"Sending request to Gemini model '{self.model_name}'...")
                # Call generate_content (non-streaming for JSON response)
                # Pass the specific response_schema needed for this call
                response = await asyncio.to_thread(
                    self.gemini_client.generate_content,
                    contents=[prompt],
                    generation_config=types.GenerationConfig(
                         response_mime_type="application/json", # Ensure JSON is requested
                         response_schema=response_schema,      # Pass the specific schema
                         temperature=0.7, # Can be overridden per call if needed
                    ),
                    # Tools are usually defined at model init, but can be passed here too
                    # tools=[types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
                )

                # --- Process Response (same logic as before) ---
                if not response.candidates or not response.candidates[0].content.parts:
                     logger.warning("Gemini API returned an empty response or no valid candidate.")
                     return None

                response_text = response.text
                logger.debug(f"Raw response text received from Gemini:\n{response_text}")

                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse Gemini JSON response: {json_err}. Response text: {response_text}")
                    return None

                logger.info(f"Gemini structured response received:\n{json.dumps(response_data, indent=2, ensure_ascii=False)}")

                # Validate required keys based on the passed schema
                if not all(key in response_data for key in response_schema.required):
                     missing_keys = [key for key in response_schema.required if key not in response_data]
                     logger.error(f"Gemini response missing required keys: {missing_keys}. Response: {response_data}")
                     return None

                return response_data # Success

            except google_exceptions.GoogleAPIError as api_err:
                # Handle potential API key errors, quota issues, etc.
                logger.error(f"Gemini API error: {api_err}")
                return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during Gemini call: {e}", exc_info=True)
                return None

    # --- run_research Method (Unchanged) ---
    async def run_research(self, question: MetaculusQuestion) -> str:
        logger.info(f"Skipping external research for {question.page_url}. Gemini uses search tool; Fallback relies on its capabilities.")
        return ""

    # --- _run_forecast_on_binary Method (Unchanged logic from v3) ---
    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Generates a forecast for a binary question using Gemini (API Key), with fallback."""
        gemini_prompt = clean_indents(f"""... [Same Gemini Prompt asking for JSON] ...""") # Keep prompt from v3
        response_data = await self._call_gemini(gemini_prompt, binary_prediction_schema)
        # --- Process Gemini / Fallback Logic (Identical to v3) ---
        if response_data:
            try:
                # ... (Extract Prediction & Rationale from JSON) ...
                prediction_value = response_data.get('Prediction')
                rationale = response_data.get('Rationale', '')
                if prediction_value is None or not isinstance(prediction_value, (float, int)): raise ValueError("Invalid Prediction")
                prediction_float = max(0.0, min(1.0, float(prediction_value)))
                logger.info(f"Gemini Forecasted URL {question.page_url} as {prediction_float:.2%}")
                return ReasonedPrediction(prediction_value=prediction_float, reasoning=rationale)
            except Exception as e: logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        if self.fallback_llm:
            logger.warning(f"Gemini failed for {question.page_url}. Using fallback {self.fallback_llm.model}...")
            fallback_prompt = clean_indents(f"""... [Same Fallback Prompt] ...""") # Keep prompt from v3
            try:
                # ... (Call fallback LLM & use PredictionExtractor) ...
                reasoning = await self.fallback_llm.invoke(fallback_prompt)
                prediction: float = PredictionExtractor.extract_last_percentage_value(reasoning, max_prediction=1, min_prediction=0)
                logger.info(f"Fallback LLM Forecasted URL {question.page_url} as {prediction:.2%}")
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
            except Exception as fallback_e: logger.error(f"Fallback LLM failed: {fallback_e}", exc_info=True)
        logger.error(f"All forecast attempts failed for binary question {question.page_url}.")
        return ReasonedPrediction(prediction_value=0.5, reasoning="Forecast generation failed.")

    # --- _run_forecast_on_multiple_choice Method (Unchanged logic from v3) ---
    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """Generates a forecast for a multiple-choice question using Gemini (API Key), with fallback."""
        gemini_prompt = clean_indents(f"""... [Same Gemini Prompt asking for JSON] ...""") # Keep prompt from v3
        response_data = await self._call_gemini(gemini_prompt, multiple_choice_prediction_schema)
        # --- Process Gemini / Fallback Logic (Identical to v3) ---
        if response_data:
            try:
                # ... (Extract Prediction list & Rationale, format/normalize) ...
                prediction_list = response_data.get('Prediction')
                rationale = response_data.get('Rationale', '')
                if not isinstance(prediction_list, list): raise ValueError("Invalid Prediction format")
                formatted_prediction: PredictedOptionList = []
                # ... (Detailed parsing/validation/normalization logic from v3) ...
                logger.info(f"Gemini Forecasted URL {question.page_url} as {formatted_prediction}")
                return ReasonedPrediction(prediction_value=formatted_prediction, reasoning=rationale)
            except Exception as e: logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        if self.fallback_llm:
            logger.warning(f"Gemini failed for {question.page_url}. Using fallback {self.fallback_llm.model}...")
            fallback_prompt = clean_indents(f"""... [Same Fallback Prompt] ...""") # Keep prompt from v3
            try:
                # ... (Call fallback LLM & use PredictionExtractor) ...
                reasoning = await self.fallback_llm.invoke(fallback_prompt)
                prediction: PredictedOptionList = PredictionExtractor.extract_option_list_with_percentage_afterwards(reasoning, question.options)
                logger.info(f"Fallback LLM Forecasted URL {question.page_url} as {prediction}")
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
            except Exception as fallback_e: logger.error(f"Fallback LLM failed: {fallback_e}", exc_info=True)
        logger.error(f"All forecast attempts failed for multiple choice question {question.page_url}.")
        num_options = len(question.options); fallback_pred = [(opt, 1.0 / num_options) for opt in question.options]
        return ReasonedPrediction(prediction_value=fallback_pred, reasoning="Forecast generation failed (even distribution).")

    # --- _run_forecast_on_numeric Method (Unchanged logic from v3) ---
    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """Generates a forecast for a numeric question using Gemini (API Key), with fallback."""
        gemini_prompt = clean_indents(f"""... [Same Gemini Prompt asking for JSON] ...""") # Keep prompt from v3
        response_data = await self._call_gemini(gemini_prompt, numeric_prediction_schema)
        # --- Process Gemini / Fallback Logic (Identical to v3) ---
        if response_data:
            try:
                # ... (Extract Prediction dict & Rationale, format) ...
                prediction_dict = response_data.get('Prediction')
                rationale = response_data.get('Rationale', '')
                if not isinstance(prediction_dict, dict): raise ValueError("Invalid Prediction format")
                # ... (Detailed percentile extraction/validation logic from v3) ...
                formatted_percentiles = {key[1:]: float(prediction_dict[key]) for key in ['p10', 'p20', 'p40', 'p60', 'p80', 'p90']}
                numeric_dist = NumericDistribution(declared_percentiles=formatted_percentiles)
                logger.info(f"Gemini Forecasted URL {question.page_url} distribution: {numeric_dist.declared_percentiles}")
                return ReasonedPrediction(prediction_value=numeric_dist, reasoning=rationale)
            except Exception as e: logger.error(f"Error processing Gemini response: {e}", exc_info=True)
        if self.fallback_llm:
            logger.warning(f"Gemini failed for {question.page_url}. Using fallback {self.fallback_llm.model}...")
            fallback_prompt = clean_indents(f"""... [Same Fallback Prompt] ...""") # Keep prompt from v3
            try:
                # ... (Call fallback LLM & use PredictionExtractor) ...
                reasoning = await self.fallback_llm.invoke(fallback_prompt)
                prediction: NumericDistribution = PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(reasoning, question)
                logger.info(f"Fallback LLM Forecasted URL {question.page_url} distribution: {prediction.declared_percentiles}")
                return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)
            except Exception as fallback_e: logger.error(f"Fallback LLM failed: {fallback_e}", exc_info=True)
        logger.error(f"All forecast attempts failed for numeric question {question.page_url}.")
        low = question.lower_bound if not question.open_lower_bound else 0; high = question.upper_bound if not question.open_upper_bound else low + 1
        if high <= low: high = low + 1
        fallback_dist = NumericDistribution(declared_percentiles={'10': low, '90': high})
        return ReasonedPrediction(prediction_value=fallback_dist, reasoning="Forecast generation failed (dummy distribution).")

    # --- _create_upper_and_lower_bound_messages Method (Unchanged) ---
    def _create_upper_and_lower_bound_messages(self, question: NumericQuestion) -> tuple[str, str]:
        # ... (Identical to v3) ...
        if question.open_upper_bound: upper_bound_message = "There is no upper bound."
        else: upper_bound_message = f"IMPORTANT: Outcome cannot be higher than {question.upper_bound}."
        if question.open_lower_bound: lower_bound_message = "There is no lower bound."
        else: lower_bound_message = f"IMPORTANT: Outcome cannot be lower than {question.lower_bound}."
        return upper_bound_message, lower_bound_message


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