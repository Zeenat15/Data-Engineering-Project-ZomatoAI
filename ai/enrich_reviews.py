
import os
import json
from pathlib import Path

import snowflake.connector
from openai import OpenAI
from dotenv import load_dotenv


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

# Always load .env from the folder containing this script.
ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# ==================================================
# 2. CONFIGURATION
# ==================================================

MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

SAMPLE_N = 5

TOPICS = [
    "food quality",
    "delivery",
    "pricing",
    "service",
    "packaging",
    "other",
]

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "ZOMATO")
SOURCE_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "RAW")
OUTPUT_SCHEMA = "AI"

SOURCE_TABLE = f"{DATABASE}.{SOURCE_SCHEMA}.REVIEWS"
OUTPUT_TABLE = f"{DATABASE}.{OUTPUT_SCHEMA}.REVIEW_ENRICHED"


# ==================================================
# 3. CONNECT TO LOCAL OLLAMA
# ==================================================

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    timeout=120.0,
    max_retries=2,
)


# ==================================================
# 4. SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = f"""
You classify customer reviews for a food delivery app.

Analyze the review and return:

1. sentiment_label:
   Exactly one of positive, negative, neutral.

2. sentiment_score:
   A number between -1.0 and 1.0.
   -1.0 = very negative
    0.0 = neutral
    1.0 = very positive

3. topic:
   Exactly one of: {", ".join(TOPICS)}.

4. key_issue:
   A short phrase of 6 words or fewer describing the
   main issue. Return null if there is no issue.

Return valid JSON only in this format:
{{
    "sentiment_label": "positive",
    "sentiment_score": 0.8,
    "topic": "food quality",
    "key_issue": null
}}

Do not include Markdown or explanations outside the JSON.
"""


# ==================================================
# 5. SNOWFLAKE CONNECTION
# ==================================================

def get_connection():
    required_vars = [
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_WAREHOUSE",
    ]

    missing_vars = [
        name for name in required_vars
        if not os.getenv(name)
    ]

    if missing_vars:
        raise ValueError(
            "Missing environment variables: "
            + ", ".join(missing_vars)
            + f". Check your .env file: {ENV_FILE}"
        )

    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=DATABASE,
        schema=SOURCE_SCHEMA,
    )


# ==================================================
# 6. CREATE OUTPUT SCHEMA AND TABLE
# ==================================================

def create_output_table(cursor):
    cursor.execute(
        f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{OUTPUT_SCHEMA}"
    )

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            REVIEW_ID STRING,
            SENTIMENT_LABEL STRING,
            SENTIMENT_SCORE FLOAT,
            TOPIC STRING,
            KEY_ISSUE STRING,
            MODEL STRING,
            ENRICHED_AT TIMESTAMP_LTZ
                DEFAULT CURRENT_TIMESTAMP()
        )
    """)


# ==================================================
# 7. FETCH UNPROCESSED REVIEWS
# ==================================================

def get_reviews_to_enrich(cursor):
    query = f"""
        SELECT r.REVIEW_ID, r.COMMENT
        FROM {SOURCE_TABLE} AS r
        WHERE r.REVIEW_ID IS NOT NULL
          AND r.COMMENT IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM {OUTPUT_TABLE} AS e
              WHERE e.REVIEW_ID = r.REVIEW_ID
          )
        LIMIT %s
    """

    cursor.execute(query, (SAMPLE_N,))
    return cursor.fetchall()


# ==================================================
# 8. CLASSIFY REVIEW USING OLLAMA
# ==================================================

def classify_review(comment):
    if not isinstance(comment, str) or not comment.strip():
        raise ValueError("Review comment is empty or invalid.")

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Classify this customer review:\n"
                    + comment
                ),
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("Ollama returned an empty response.")

    labels = json.loads(answer)

    required_keys = {
        "sentiment_label",
        "sentiment_score",
        "topic",
        "key_issue",
    }

    if not isinstance(labels, dict):
        raise ValueError("Model response must be a JSON object.")

    if not required_keys.issubset(labels):
        raise ValueError(
            f"Missing output fields: {required_keys - labels.keys()}"
        )

    # Validate sentiment label.
    sentiment = labels["sentiment_label"]

    if not isinstance(sentiment, str):
        raise ValueError("Invalid sentiment_label.")

    sentiment = sentiment.strip().lower()

    if sentiment not in {"positive", "negative", "neutral"}:
        raise ValueError(f"Invalid sentiment: {sentiment}")

    # Validate sentiment score.
    score = labels["sentiment_score"]

    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Sentiment score must be numeric.")

    score = float(score)

    if not -1.0 <= score <= 1.0:
        raise ValueError("Sentiment score must be between -1 and 1.")

    # Validate topic.
    topic = labels["topic"]

    if not isinstance(topic, str):
        raise ValueError("Invalid topic.")

    topic = topic.strip().lower()

    if topic not in TOPICS:
        raise ValueError(f"Invalid topic: {topic}")

    # Validate key issue.
    key_issue = labels["key_issue"]

    if key_issue is not None:
        if not isinstance(key_issue, str):
            raise ValueError("key_issue must be a string or null.")

        key_issue = " ".join(key_issue.strip().split())

        if not key_issue:
            key_issue = None
        else:
            key_issue = " ".join(key_issue.split()[:6])

    return {
        "sentiment_label": sentiment,
        "sentiment_score": score,
        "topic": topic,
        "key_issue": key_issue,
    }


# ==================================================
# 9. SAVE RESULTS TO SNOWFLAKE
# ==================================================

def save_results(cursor, results):
    if not results:
        print("No successfully classified reviews to save.")
        return

    print(f"Saving {len(results)} reviews to Snowflake...")

    # Recheck for existing IDs before each insert.
    # This avoids ordinary repeat-run duplicates, although
    # concurrent script executions can still race.
    insert_query = f"""
        INSERT INTO {OUTPUT_TABLE} (
            REVIEW_ID,
            SENTIMENT_LABEL,
            SENTIMENT_SCORE,
            TOPIC,
            KEY_ISSUE,
            MODEL
        )
        SELECT %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (
            SELECT 1
            FROM {OUTPUT_TABLE}
            WHERE REVIEW_ID = %s
        )
    """

    for row in results:
        cursor.execute(insert_query, (*row, row[0]))


# ==================================================
# 10. MAIN PIPELINE
# ==================================================

def main():
    conn = None
    cursor = None

    try:
        print("Connecting to Snowflake...")

        conn = get_connection()
        cursor = conn.cursor()

        print("Connected to Snowflake successfully.")

        print(f"Source table: {SOURCE_TABLE}")
        print(f"Output table: {OUTPUT_TABLE}")

        create_output_table(cursor)
        conn.commit()

        print("Fetching unprocessed reviews...")

        reviews = get_reviews_to_enrich(cursor)

        if not reviews:
            print("No new reviews to enrich.")
            return

        print(f"Found {len(reviews)} reviews to process.")

        results = []

        for review_id, comment in reviews:
            print(f"\nClassifying review ID: {review_id}")
            print(f"Review: {comment}")

            try:
                labels = classify_review(comment)

                print("Classification:", labels)

                results.append((
                    review_id,
                    labels["sentiment_label"],
                    labels["sentiment_score"],
                    labels["topic"],
                    labels["key_issue"],
                    MODEL,
                ))

            except Exception as exc:
                print(
                    f"Classification failed for review "
                    f"{review_id}: {exc}"
                )

        if results:
            save_results(cursor, results)
            conn.commit()

            print(
                f"\nPipeline completed. "
                f"{len(results)} reviews classified successfully."
            )
            print(f"Results stored in {OUTPUT_TABLE}.")

        else:
            print(
                "\nNo reviews were successfully classified. "
                "Nothing was inserted."
            )

    except Exception as exc:
        if conn is not None:
            conn.rollback()

        print(f"\nPipeline failed: {exc}")
        raise

    finally:
        if cursor is not None:
            cursor.close()

        if conn is not None:
            conn.close()

        print("Snowflake connection closed.")


# ==================================================
# 11. ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
