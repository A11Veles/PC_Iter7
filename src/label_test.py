import pandas as pd
import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Set
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
API_KEY = "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"
MODEL_NAME = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.1
MAX_TOKENS = 100

# Enhanced configuration for reliability
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 300    # 5 minutes
RATE_LIMIT_DELAY = 60    # 1 minute base delay for rate limits
REQUEST_TIMEOUT = 30     # 30 seconds timeout

# File paths
PROCESSED_IDS_FILE = "processed_ids.json"
LOG_FILE = "classification_log.txt"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prompt template
PROMPT_TEMPLATE = """
You are a product classification expert. Given a product name, classify it into the most appropriate category from the provided list.

Product Name: {{Product_Name}}

Available Categories:
{{LABEL_1}}
{{LABEL_2}}
{{LABEL_3}}
{{LABEL_4}}
{{LABEL_5}}
....

Instructions:
- Return ONLY the exact category name from the list above
- Choose the most specific and accurate category
- Do not add any explanation or additional text

Category:
"""

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_processed_ids() -> Set[int]:
    """Load previously processed product IDs"""
    if os.path.exists(PROCESSED_IDS_FILE):
        try:
            with open(PROCESSED_IDS_FILE, 'r') as f:
                processed_data = json.load(f)
                processed_ids = set(processed_data.get('processed_ids', []))
                logger.info(f"Loaded {len(processed_ids)} previously processed IDs")
                return processed_ids
        except Exception as e:
            logger.error(f"Error loading processed IDs: {e}")
            return set()
    else:
        logger.info("No previous processing history found")
        return set()

def save_processed_ids(processed_ids: Set[int]):
    """Save processed product IDs to file"""
    try:
        processed_data = {
            'processed_ids': list(processed_ids),
            'last_updated': datetime.now().isoformat(),
            'total_processed': len(processed_ids)
        }
        with open(PROCESSED_IDS_FILE, 'w') as f:
            json.dump(processed_data, f, indent=2)
        logger.info(f"Saved {len(processed_ids)} processed IDs to {PROCESSED_IDS_FILE}")
    except Exception as e:
        logger.error(f"Error saving processed IDs: {e}")

def get_timestamp_suffix() -> str:
    """Generate timestamp suffix for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_data(csv_file_path):
    """Load the CSV data"""
    df = pd.read_csv(csv_file_path)
    return df

def get_unique_labels(df, column_name):
    """Get unique labels from a column"""
    return df[column_name].dropna().unique().tolist()

def prepare_prompt(product_name, allowed_labels):
    """Prepare the prompt for LLM"""
    labels_text = "\n".join(allowed_labels)
    return PROMPT_TEMPLATE.replace("{{Product_Name}}", product_name).replace("{{LABEL_1}}\n{{LABEL_2}}\n{{LABEL_3}}\n{{LABEL_4}}\n{{LABEL_5}}\n....", labels_text)

def make_api_request_with_retry(prompt, session=None):
    """Make API request to LLM with exponential backoff retry logic"""
    if session is None:
        session = create_session_with_retries()
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    
    retry_delay = INITIAL_RETRY_DELAY
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"API request attempt {attempt + 1}/{MAX_RETRIES}")
            
            response = session.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            
            # Handle different HTTP status codes
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                else:
                    logger.warning("Empty response from API")
                    return ""
                    
            elif response.status_code == 429:  # Rate limit
                logger.warning(f"Rate limit hit (429). Waiting {RATE_LIMIT_DELAY} seconds...")
                time.sleep(RATE_LIMIT_DELAY)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
                
            elif response.status_code in [500, 502, 503, 504]:  # Server errors
                logger.warning(f"Server error {response.status_code}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
                continue
                
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            if attempt == MAX_RETRIES - 1:
                return ""
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt == MAX_RETRIES - 1:
                return ""
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
    
    logger.error(f"All {MAX_RETRIES} attempts failed")
    return ""

def parse_response(response, allowed_labels):
    """Parse LLM response to extract the predicted label"""
    if not response:
        return allowed_labels[0] if allowed_labels else ""
    
    response = response.strip()
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    # Check for exact matches first
    for line in lines:
        line_clean = line.replace('```', '').strip()
        if line_clean in allowed_labels:
            return line_clean
    
    # Check for partial matches
    for label in allowed_labels:
        if label.lower() in response.lower():
            return label
    
    # Return first line if no match found
    if lines:
        return lines[0].replace('```', '').strip()
    
    return allowed_labels[0] if allowed_labels else ""

def classify_hierarchical(product_name, df_gpc, session):
    """Classify product through all 4 GPC levels with error handling"""
    logger.info(f"Classifying: {product_name}")
    
    try:
        # Level 1: Segment
        segments = get_unique_labels(df_gpc, 'gpc_segment')
        segment_prompt = prepare_prompt(product_name, segments)
        segment_response = make_api_request_with_retry(segment_prompt, session)
        predicted_segment = parse_response(segment_response, segments)
        logger.info(f"  Segment: {predicted_segment}")
        
        # Level 2: Family (filter by predicted segment)
        segment_filtered = df_gpc[df_gpc['gpc_segment'] == predicted_segment]
        families = get_unique_labels(segment_filtered, 'gpc_family')
        if families:
            family_prompt = prepare_prompt(product_name, families)
            family_response = make_api_request_with_retry(family_prompt, session)
            predicted_family = parse_response(family_response, families)
        else:
            predicted_family = ""
        logger.info(f"  Family: {predicted_family}")
        
        # Level 3: Class (filter by predicted segment and family)
        family_filtered = segment_filtered[segment_filtered['gpc_family'] == predicted_family]
        classes = get_unique_labels(family_filtered, 'gpc_class')
        if classes:
            class_prompt = prepare_prompt(product_name, classes)
            class_response = make_api_request_with_retry(class_prompt, session)
            predicted_class = parse_response(class_response, classes)
        else:
            predicted_class = ""
        logger.info(f"  Class: {predicted_class}")
        
        # Level 4: Brick (filter by predicted segment, family, and class)
        class_filtered = family_filtered[family_filtered['gpc_class'] == predicted_class]
        bricks = get_unique_labels(class_filtered, 'gpc_brick')
        if bricks:
            brick_prompt = prepare_prompt(product_name, bricks)
            brick_response = make_api_request_with_retry(brick_prompt, session)
            predicted_brick = parse_response(brick_response, bricks)
        else:
            predicted_brick = ""
        logger.info(f"  Brick: {predicted_brick}")
        
        # Rate limiting between requests
        time.sleep(0.5)
        
        return {
            'predicted_segment': predicted_segment,
            'predicted_family': predicted_family,
            'predicted_class': predicted_class,
            'predicted_brick': predicted_brick
        }
        
    except Exception as e:
        logger.error(f"Error in classify_hierarchical for '{product_name}': {e}")
        # Return empty predictions on error
        return {
            'predicted_segment': "",
            'predicted_family': "",
            'predicted_class': "",
            'predicted_brick': ""
        }

def filter_unprocessed_data(df: pd.DataFrame, processed_ids: Set[int], batch_size: int = 100) -> pd.DataFrame:
    """Filter out already processed items and return next batch"""
    # Filter out processed items
    unprocessed_df = df[~df['id'].isin(processed_ids)]
    
    logger.info(f"Total items in dataset: {len(df)}")
    logger.info(f"Already processed: {len(processed_ids)}")
    logger.info(f"Remaining unprocessed: {len(unprocessed_df)}")
    
    if len(unprocessed_df) == 0:
        logger.info("All items have been processed!")
        return pd.DataFrame()
    
    # Take next batch
    batch_df = unprocessed_df.head(batch_size)
    logger.info(f"Processing next batch of {len(batch_df)} items")
    
    return batch_df

def process_batch(df_batch, df_gpc_reference, processed_ids, batch_number):
    """Process a single batch and return results with enhanced error handling"""
    results = []
    newly_processed_ids = set()
    session = create_session_with_retries()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING BATCH #{batch_number}")
    logger.info(f"{'='*60}")
    logger.info(f"Processing {len(df_batch)} products...")
    
    for idx, row in df_batch.iterrows():
        try:
            product_name = row['translated_name']
            product_id = row['id']
            
            logger.info(f"\n[{len(results)+1}/{len(df_batch)}] Processing ID {product_id}: {product_name[:50]}...")
            
            # Get LLM predictions
            predictions = classify_hierarchical(product_name, df_gpc_reference, session)
            
            # Create result with only predictions (no comparison)
            result = {
                'id': product_id,
                'translated_name': product_name,
                'predicted_segment': predictions['predicted_segment'],
                'predicted_family': predictions['predicted_family'],
                'predicted_class': predictions['predicted_class'],
                'predicted_brick': predictions['predicted_brick']
            }
            results.append(result)
            newly_processed_ids.add(product_id)
            
            # Save progress every 5 items (more frequent saves)
            if len(results) % 5 == 0:
                updated_processed_ids = processed_ids.union(newly_processed_ids)
                save_processed_ids(updated_processed_ids)
                logger.info(f"Progress saved: {len(results)} items completed in this batch")
                
        except Exception as e:
            logger.error(f"Error processing product ID {product_id}: {e}")
            # Continue with next item instead of stopping
            continue
    
    if not results:
        logger.warning("No results generated for this batch")
        return newly_processed_ids
    
    try:
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Generate timestamped filename
        timestamp = get_timestamp_suffix()
        predictions_filename = f'gpc_predictions_batch{batch_number}_{timestamp}.csv'
        
        # Save predictions results
        results_df.to_csv(predictions_filename, index=False)
        logger.info(f"Saved {predictions_filename}")
        
        logger.info(f"\n=== BATCH #{batch_number} COMPLETED ===")
        logger.info(f"Successfully processed {len(results_df)} items")
            
    except Exception as e:
        logger.error(f"Error saving batch results: {e}")
    
    return newly_processed_ids

def main():
    logger.info("🚀 Starting continuous batch processing...")
    logger.info("Press Ctrl+C to stop at any time")
    logger.info("="*60)
    
    # Load processed IDs
    processed_ids = load_processed_ids()
    
    # Load your data to classify
    logger.info("Loading data to classify...")
    try:
        df = pd.read_csv('data/translated_test_dataset.csv')
        logger.info(f"Loaded dataset with {len(df)} items")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Load reference GPC data for classification hierarchy
    logger.info("Loading GPC reference data...")
    try:
        df_gpc_reference = pd.read_csv('data/usda_to_gpc_after.csv')
        logger.info(f"Loaded GPC reference data with {len(df_gpc_reference)} items")
    except Exception as e:
        logger.error(f"Error loading GPC reference data: {e}")
        return
    
    batch_size = 50  # Reduced batch size for better error recovery
    batch_number = 1
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    try:
        while True:
            try:
                # Filter unprocessed data and get next batch
                df_batch = filter_unprocessed_data(df, processed_ids, batch_size)
                
                if df_batch.empty:
                    logger.info("\n🎉 ALL ITEMS HAVE BEEN PROCESSED!")
                    logger.info("No more items to process. Exiting.")
                    break
                
                # Process the batch
                newly_processed_ids = process_batch(df_batch, df_gpc_reference, processed_ids, batch_number)
                
                # Update processed IDs
                processed_ids = processed_ids.union(newly_processed_ids)
                save_processed_ids(processed_ids)
                
                logger.info(f"\n=== BATCH #{batch_number} SUMMARY ===")
                logger.info(f"Items processed in this batch: {len(newly_processed_ids)}")
                logger.info(f"Total items processed so far: {len(processed_ids)}")
                logger.info(f"Remaining items in dataset: {len(df) - len(processed_ids)}")
                
                batch_number += 1
                consecutive_failures = 0  # Reset failure counter on success
                
                # Small delay before next batch
                logger.info(f"\nMoving to next batch in 5 seconds...")
                time.sleep(5)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error in batch {batch_number}: {e}")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Too many consecutive failures ({consecutive_failures}). Stopping.")
                    break
                
                logger.info(f"Waiting 30 seconds before retrying... (Failure {consecutive_failures}/{max_consecutive_failures})")
                time.sleep(30)
                
    except KeyboardInterrupt:
        logger.info(f"\n\n⏹️  STOPPED BY USER (Ctrl+C)")
        logger.info(f"Final processing summary:")
        logger.info(f"- Completed batches: {batch_number - 1}")
        logger.info(f"- Total items processed: {len(processed_ids)}")
        logger.info(f"- Remaining items: {len(df) - len(processed_ids)}")
        logger.info(f"- Progress saved to: {PROCESSED_IDS_FILE}")
        logger.info("You can resume processing by running the script again.")
    
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.info(f"Progress has been saved. You can resume by running the script again.")

if __name__ == "__main__":
    main()