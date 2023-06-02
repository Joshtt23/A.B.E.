import unittest
import importlib
import glob
import time
from main import LoadingBar  # Import the LoadingBar class from root/main.py

def run_tests():
    test_modules = glob.glob('tnt/**/test.py', recursive=True)
    total_models = len(test_modules)
    loading_bar = LoadingBar(total=total_models)  # Create a loading bar with total_models

    for i, test_module in enumerate(test_modules):
        try:
            module_name = test_module.replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_name)
            if hasattr(module, 'run_tests'):
                loading_bar.update_progress(1, f"Testing Model {i + 1}/{total_models}")  # Update the loading bar for each model
                module.run_tests()
            else:
                raise AttributeError(f"Module '{module_name}' does not have a 'run_tests' function")
        except ImportError:
            print(f"Error importing test module: {module_name}")
        except Exception as e:
            print(f"Error running tests for module: {module_name}")
            print(str(e))
        print()

    loading_bar.reset()  # Reset the loading bar


if __name__ == '__main__':
    run_tests()
