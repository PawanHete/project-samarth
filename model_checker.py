import google.generativeai as genai
import os
from dotenv import load_dotenv

def check_available_models():
    """
    Connects to the Google AI API and lists all available models
    that support the 'generateContent' method.
    """
    try:
        # Load the API key from the .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            print("❌ Error: GOOGLE_API_KEY not found in your .env file.")
            return

        # Configure the genai library with the API key
        genai.configure(api_key=api_key)

        print("Fetching available models from Google AI...\n")
        
        found_models = False
        # List all models and check if they support the required method
        for m in genai.list_models():
          if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Found model: {m.name}")
            found_models = True
        
        if not found_models:
            print("❌ No models supporting 'generateContent' were found for your API key.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    check_available_models()

### **How to Use This Script**

