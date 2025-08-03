import pandas as pd
import openai
import time

# CONFIG
openai.api_key = "YOUR_OPENAI_API_KEY"
MODEL = "gpt-3.5-turbo"
OUTPUT_PATH = "gen_data_with_intent.csv"

# Load CSVs
general = pd.read_csv("chats_general.csv")
specific = pd.read_csv("chats_specific.csv")

# Merge and shuffle
combined = pd.concat([general, specific]).sample(frac=1, random_state=42).reset_index(drop=True)
gen_data = combined[combined["content_type"] == "prompt"].copy()
gen_data["predicted_intent"] = ""

# Prediction function
def predict_intent(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that classifies user prompts into one of three intent categories: "
                    "consideration, evaluation, or decision. Only return the category name."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error: {e}")
        return "error"

# Loop over prompts
for i, row in gen_data.iterrows():
    gen_data.at[i, "predicted_intent"] = predict_intent(row["content"])

    if (i + 1) % 10 == 0:
        print(f"{i + 1} prompts processed...")
    
    time.sleep(0.5)  # avoid rate limits

# Save output
gen_data.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved to {OUTPUT_PATH}")
