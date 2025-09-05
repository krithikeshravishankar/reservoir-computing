from flask import Flask, render_template

# Create an instance of the Flask class
app = Flask(__name__)

# Define a "route" that maps a URL to a Python function
@app.route('/')
def home():
    """This function runs when someone visits the root URL ('/')"""
    # The render_template function finds and sends the HTML file
    return render_template('index.html')

# The following is needed to run the app with `python app.py`
if __name__ == '__main__':
    app.run(debug=True)