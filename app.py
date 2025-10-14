from flask import Flask, render_template, url_for, redirect, request
import os
import time

# Import the refactored scripts
from notebooks import run_reservoir_forecast as forecast_runner
from notebooks import plot_forecast as forecast_plotter

# Create an instance of the Flask class
app = Flask(__name__)

# Define a "route" that maps a URL to a Python function
@app.route('/', methods=['GET', 'POST'])
def home():
    """This function runs when someone visits the root URL ('/')"""
    if request.method == 'POST':
        # Define the configuration for the forecast
        config = forecast_runner.Config(
            system_name="lorenz",
            forecast_time=25.0,
            training_time=100.0,
            seed=int(time.time())  # Use a new seed each time
        )
        # Run the forecast pipeline
        results = forecast_runner.run(config)

        # Define a unique filename base relative to the static folder
        filename_base = f'plots/forecast_{config.seed}'
        # Get the absolute path for saving the files
        save_path_base = os.path.join(app.static_folder, filename_base)
        
        # Generate and save the plots to the absolute path
        forecast_plotter.save_plots_to_file(save_path_base, results)

        # Create a dictionary of relative paths for the template's url_for function
        plot_paths = {
            "3d": f"{filename_base}_3d.png",
            "components": {
                "X": f"{filename_base}_comp_x.png",
                "Y": f"{filename_base}_comp_y.png",
                "Z": f"{filename_base}_comp_z.png",
            }
        }
        return render_template('index.html', plots=plot_paths)

    return render_template('index.html', plots=None)

# The following is needed to run the app with `python app.py`
if __name__ == '__main__':
    app.run(debug=True)