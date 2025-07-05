import openai
import csv
import uuid
import random
from datetime import datetime, timedelta
import time

# CONFIG
openai.api_key = "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"  # Replace with your actual key
NUM_RECORDS = 700  # Adjust as needed
MODEL = "gpt-3.5-turbo"

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai.api_key)

# Prompt instruction (specific version)
PROMPT_INSTRUCTION = (
    "Generate a realistic car-related question from 2025 that does not mention any specific brand or model, focused on buying advice, EVs, features, or lifestyle needs, in a natural tone."
)

# FILE SETUP
chats_path = "chats_general.csv"
actions_path = "actions_general.csv"

# INIT CSV FILES
with open(chats_path, "w", newline='', encoding='utf-8') as chat_file, \
     open(actions_path, "w", newline='', encoding='utf-8') as action_file:

    chat_writer = csv.DictWriter(chat_file, fieldnames=[
        "content_type", "content", "created_at", "content_id",
        "machine_id", "chat_group_id", "sources"
    ])
    action_writer = csv.DictWriter(action_file, fieldnames=[
        "action_type", "created_at", "content_id"
    ])

    chat_writer.writeheader()
    action_writer.writeheader()

    for i in range(NUM_RECORDS):
        machine_id = str(uuid.uuid4())
        chat_group_id = str(uuid.uuid4())

        # Ensure all dates fall between June 1 and June 30, 2025
        base_time = datetime(2025, 6, 1)
        end_time = datetime(2025, 6, 30, 23, 59, 59)
        prompt_time = base_time + timedelta(seconds=random.randint(0, int((end_time - base_time).total_seconds() - 180)))
        now = prompt_time.isoformat()
        response_dt = prompt_time + timedelta(seconds=random.randint(30, 120))
        response_time = response_dt.isoformat()
        action_time = (response_dt + timedelta(seconds=random.randint(10, 60))).isoformat()

        try:
            # Generate user prompt
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": PROMPT_INSTRUCTION}]
            )
            user_prompt = response.choices[0].message.content.strip()

            # Generate AI response to the prompt
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_prompt}]
            )
            ai_response = completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"❌ Error at record {i + 1}: {e}")
            continue

        # Fake URL sources in response
        fake_urls = [
            "https://autotrends.org/2025-electric-guide",
            "https://cars.ai/news/top-picks",
            "https://carbuzz.net/battery-comparison",
            "https://reviewauto2025.com/hybrids"
        ]
        sources = ";".join(random.sample(fake_urls, k=random.randint(0, 2)))

        # Unique IDs
        prompt_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())

        # Write prompt
        chat_writer.writerow({
            "content_type": "prompt",
            "content": user_prompt,
            "created_at": now,
            "content_id": prompt_id,
            "machine_id": machine_id,
            "chat_group_id": chat_group_id,
            "sources": ""
        })

        time.sleep(0.5)  # Slight delay to reduce rate-limiting

        # Write response
        chat_writer.writerow({
            "content_type": "response",
            "content": ai_response,
            "created_at": response_time,
            "content_id": response_id,
            "machine_id": machine_id,
            "chat_group_id": chat_group_id,
            "sources": sources
        })

        # Simulate an action (clicked or purchased)
        action_writer.writerow({
            "action_type": random.choice(["clicked", "purchased"]),
            "created_at": action_time,
            "content_id": response_id
        })

        # Print progress every 100 records
        if (i + 1) % 5 == 0:
            print(f"✅ Generated {i + 1} records...")

print("🎉 CSV generation complete.")
