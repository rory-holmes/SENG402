import unittest
import os
import sys
sys.path.append('params')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.global_helpers as gh
import yaml

with open(r"params\\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\feature_model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)


def createFakeHistoryFiles():
    """
    Creates two false train-history files for testing
    """
    file1_name = os.path.join(params['results_path'], 'InceptionResNetV2(900)_training-history')
    file2_name = os.path.join(params['results_path'], 'InceptionResNetV2(901)_training-history')
    with open(file1_name, 'w') as file1:
        file1.writelines("This is a test")
    with open(file2_name, 'w') as file2:
        file2.writelines("This is a test")

def tearDownTestSuite():
    """
    Delete all files within and the 'test_suite' dir
    """
    if os.path.exists(params['results_path']):
        for filename in os.listdir(params['results_path']):
            if 'InceptionResNetV2(90' in filename:
                file_path = os.path.join(params['results_path'], filename)
                os.unlink(file_path)

class TestSringMethods(unittest.TestCase):
    def test_givenModelName_whenGetLoggerName_thenCorrectNameReturned(self):
        createFakeHistoryFiles()
        new_name = gh.get_logger_name("InceptionResNetV2")
        self.assertEqual("InceptionResNetV2(902)_training-history", new_name)
        tearDownTestSuite()


if __name__ == '__main__':
    unittest.main()
