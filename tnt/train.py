import unittest
import importlib
import glob
import time
from main import LoadingBar  # Import the LoadingBar class from root/main.py

def run_training():
    train_scripts = glob.glob('tnt/**/train.py', recursive=True)
    total_models = len(train_scripts)
    loading_bar = LoadingBar(total=total_models)  # Create a loading bar with total_models

    for i, train_script in enumerate(train_scripts):
        try:
            module_name = train_script.replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_name)
            if hasattr(module, 'run_training'):
                loading_bar.update_progress(1, f"Training Model {i + 1}/{total_models}")  # Update the loading bar for each model
                module.run_training()
            else:
                raise AttributeError(f"Module '{module_name}' does not have a 'run_training' function")
        except ImportError:
            raise ImportError(f"Error importing '{module_name}' module")

    loading_bar.reset()  # Reset the loading bar


if __name__ == '__main__':
    run_training()
