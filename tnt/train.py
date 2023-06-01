import unittest
import importlib
import glob

class TrainTest(unittest.TestCase):
    def test_train_scripts(self):
        train_scripts = glob.glob('**/train.py', recursive=True)
        for train_script in train_scripts:
            try:
                train_module = importlib.import_module(train_script.replace('.py', '').replace('/', '.'))
                self.assertTrue(hasattr(train_module, 'run_training'))
            except ImportError:
                self.fail(f"Error importing '{train_script}' module")
    
if __name__ == '__main__':
    unittest.main()
