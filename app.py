from flask import Flask, render_template, request

# Create an instance of the Flask class
app = Flask(__name__)

def predict_sentiment(text):
    """
    A simple mock ML model function for sentiment analysis.
    In a real application, this would be a call to a trained model.
    """
    text = text.lower()
    positive_words = ['good', 'great', 'excellent', 'happy', 'love', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate', 'worst']

    if any(word in text for word in positive_words):
        return "Positive"
    elif any(word in text for word in negative_words):
        return "Negative"
    else:
        return "Neutral"

# Define a "route" that maps a URL to a Python function
@app.route('/', methods=['GET', 'POST'])
def home():
    """This function runs when someone visits the root URL ('/')"""
    if request.method == 'POST':
        text_from_form = request.form['user_text']
        # Feed the text to our "model"
        prediction = predict_sentiment(text_from_form)
        # Re-render the page, passing both the original text and the prediction
        return render_template('index.html', submitted_text=text_from_form, prediction=prediction)

    return render_template('index.html', submitted_text=None, prediction=None)

# The following is needed to run the app with `python app.py`
if __name__ == '__main__':
    app.run(debug=True)