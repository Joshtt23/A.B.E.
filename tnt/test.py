import importlib
import sys
import unittest
import glob


def run_tests():
    # Find all test modules
    test_modules = glob.glob('**/test.py', recursive=True)

    # Import and run tests for each module
    for test_module in test_modules:
        try:
            module = importlib.import_module(test_module.replace('.py', '').replace('/', '.'))
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            unittest.TextTestRunner().run(suite)
        except ImportError:
            print(f"Error importing test module: {test_module}")
        except Exception as e:
            print(f"Error running tests for module: {test_module}")
            print(str(e))
        print()


if __name__ == '__main__':
    run_tests()
