from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("Energy_efficiency.pkl")

# Original feature names (exactly as used in training)
model_features = list(model.feature_names_in_)

# HTML-safe feature names (replace spaces with underscore)
html_features = [f.replace(" ", "_") for f in model_features]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        input_data = {}

        # Map HTML names back to original model feature names
        for original, safe in zip(model_features, html_features):
            value = request.form.get(safe)

            if value is None or value.strip() == "":
                value = 0

            input_data[original] = [float(value)]

        input_df = pd.DataFrame(input_data)

        prediction = model.predict(input_df)[0]

    return render_template(
        "index.html",
        features=html_features,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
