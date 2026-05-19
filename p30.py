import streamlit as st
import numpy as np
import requests
import json
import os

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(layout="centered")

# ==============================
# OPENROUTER API CONFIGURATION
# ==============================

API_KEY = os.getenv("OPENROUTER_API_KEY", "")


# ==============================
# APP TITLE
# ==============================

st.title("🐄 Veterinary Decision Support App")

st.write("Enter milk production and health details for analysis.")

# ==============================
# SIDEBAR SETTINGS
# ==============================

st.sidebar.title("Farm Settings")

cow_count = st.sidebar.number_input(
    "Enter Number of Cows",
    min_value=1,
    max_value=50,
    value=3
)

# ==============================
# DATA STORAGE
# ==============================

milk_values = []
temp_values = []
feed_values = []
cow_names = []

# ==============================
# INPUT SECTION
# ==============================

st.subheader("Milk Production Data")

for i in range(cow_count):

    st.markdown(f"## Cow {i+1}")

    cow_name = st.text_input(
        f"Enter Cow {i+1} Name",
        key=f"name_{i}"
    )

    milk = st.number_input(
        f"Cow {i+1} Milk Production (Liters)",
        min_value=0.0,
        step=0.5,
        key=f"milk_{i}"
    )

    temp = st.number_input(
        f"Cow {i+1} Body Temperature (°F)",
        min_value=90.0,
        max_value=110.0,
        step=0.1,
        value=101.5,
        key=f"temp_{i}"
    )

    feed = st.slider(
        f"Cow {i+1} Feed Intake",
        0,
        100,
        50,
        key=f"feed_{i}"
    )

    # Store Values

    cow_names.append(cow_name if cow_name else f"Cow {i+1}")

    milk_values.append(milk)

    temp_values.append(temp)

    feed_values.append(feed)

# ==============================
# ANALYSIS BUTTON
# ==============================

if st.button("Analyze Health"):

    y = np.array(milk_values)

    avg_milk = np.mean(y)

    st.subheader("Health Analysis Result")

    st.write(f"### Average Milk Production: {avg_milk:.2f} Liters")

    # ==========================
    # INDIVIDUAL COW ANALYSIS
    # ==========================

    for i, milk in enumerate(y):

        st.markdown("---")

        st.write(f"## {cow_names[i]}")

        st.write(f"Milk Production: {milk} Liters")

        st.write(f"Body Temperature: {temp_values[i]} °F")

        st.write(f"Feed Intake: {feed_values[i]}")

        # Severe Issue

        if milk < avg_milk * 0.5:

            st.error(f"{cow_names[i]}: Severe milk drop detected")

            st.write("### Possible Causes")

            st.write("- Mastitis risk")

            st.write("- Nutrition deficiency")

            st.write("- Serious infection")

        # Moderate Issue

        elif milk < avg_milk * 0.8:

            st.warning(f"{cow_names[i]}: Moderate milk reduction detected")

            st.write("### Possible Causes")

            st.write("- Early infection")

            st.write("- Heat stress")

            st.write("- Feed imbalance")

        # Normal

        else:

            st.success(f"{cow_names[i]}: Normal milk production")

        # Temperature Check

        if temp_values[i] > 103:

            st.warning(f"{cow_names[i]} has high fever.")

        # Feed Check

        if feed_values[i] < 30:

            st.warning(f"{cow_names[i]} has low feed intake.")

    # ==========================
    # FINAL SUMMARY
    # ==========================

    st.markdown("---")

    st.subheader("Final Summary")

    if min(y) < avg_milk * 0.5:

        st.error("⚠️ High-risk animal detected in the farm.")

    elif min(y) < avg_milk * 0.8:

        st.warning("⚠️ Some cows need monitoring.")

    else:

        st.success("✅ Farm condition looks normal.")

# ==============================
# AI VETERINARY ASSISTANT
# ==============================

st.markdown("---")

st.subheader("🤖 AI Veterinary Assistant")

user_problem = st.text_area(
    "Describe the animal problem"
)

if st.button("Analysis with AI"):

    if user_problem.strip() == "":

        st.warning("Please describe the animal problem.")

    else:

        try:

            prompt = f"""
            You are an expert veterinary assistant.

            Analyze the following animal health issue and provide:

            1. Possible causes
            2. Suggested precautions
            3. Recommended treatment ideas
            4. Whether veterinary attention is urgently needed

            Animal Problem:
            {user_problem}
            """

            # ==============================
            # FIRST API CALL WITH REASONING
            # ==============================

            response = requests.post(

                url="https://openrouter.ai/api/v1/chat/completions",

                headers={

                    "Authorization": f"Bearer {API_KEY}",

                    "Content-Type": "application/json",

                },

                data=json.dumps({

                    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",

                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    "reasoning": {
                        "enabled": True
                    }

                })

            )

            # ==============================
            # RESPONSE CONVERSION
            # ==============================

            response_data = response.json()

            # ==============================
            # RESPONSE EXTRACTION
            # ==============================

            assistant_message = response_data['choices'][0]['message']

            # ==============================
            # PRESERVE REASONING DETAILS
            # ==============================

            messages = [

                {
                    "role": "user",
                    "content": prompt
                },

                {
                    "role": "assistant",
                    "content": assistant_message.get("content"),

                    "reasoning_details":
                    assistant_message.get("reasoning_details")
                },

                {
                    "role": "user",
                    "content": "Think carefully and provide the final veterinary analysis."
                }

            ]

            # ==============================
            # SECOND API CALL
            # ==============================

            response2 = requests.post(

                url="https://openrouter.ai/api/v1/chat/completions",

                headers={

                    "Authorization": f"Bearer {API_KEY}",

                    "Content-Type": "application/json",

                },

                data=json.dumps({

                    "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",

                    "messages": messages,

                    "reasoning": {
                        "enabled": True
                    }

                })

            )

            # ==============================
            # FINAL RESPONSE
            # ==============================

            final_result = response2.json()

            final_reply = final_result['choices'][0]['message']['content']

            # ==============================
            # SHOW RESULT
            # ==============================

            st.subheader("📋 AI Analysis Result")

            st.write(final_reply)

        except Exception as e:

            st.error(f"API Error: {e}")
 


