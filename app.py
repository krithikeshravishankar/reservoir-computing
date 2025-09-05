from flask import Flask, render_template, request

# Create an instance of the Flask class
app = Flask(__name__)

# Define a "route" that maps a URL to a Python function
@app.route('/', methods=['GET', 'POST'])
def home():
    """This function runs when someone visits the root URL ('/')"""
    
    # Check if the request is a POST request (i.e., the form was submitted)
    if request.method == 'POST':
        # Get the data from the form input field named 'user_text'
        text_from_form = request.form['user_text']
        print(f"Received from form: {text_from_form}")
        # Re-render the page, passing the received text back to the template
        return render_template('index.html', submitted_text=text_from_form)

    # If it's a GET request (the user just loaded the page), render it normally
    return render_template('index.html', submitted_text=None)

# The following is needed to run the app with `python app.py`
if __name__ == '__main__':
    app.run(debug=True)