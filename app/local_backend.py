from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = "Mechabruh/retrained_model"  # Replace with your Hugging Face model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/translate", methods=["POST"])
def translate():
    # Get input text from the request
    data = request.json
    input_text = data.get("text")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)

    # Generate translation
    outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)

    # Decode output tokens
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translation": translated_text})

if __name__ == "__main__":
    app.run(debug=True)