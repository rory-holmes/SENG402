import unittest
import os
import sys
sys.path.append('params')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.file_helpers as fh
import yaml

with open(r"params\\paths.yaml", "r") as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\feature_model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

def createFakeFiles(path):
    """
    Creates 4 false files for testing purposes
    """
    files = []
    for i in range(0,4):
        files.append(os.path.join(path, f"ThisIsAFakeFile{i}"))

    for file in files:
        with open(file, 'w') as file1:
            file1.writelines("This is a test")

def tearDownTestSuite(path):
    """
    Delete all files within and the 'test_suite' dir
    """
    if os.path.exists(path):
        for filename in os.listdir(path):
            if 'ThisIsAFakeFile' in filename:
                file_path = os.path.join(path, filename)
                os.unlink(file_path)

class TestSringMethods(unittest.TestCase):
    def test_givenData_whenSplitData_thenDataSplitCorrectly(self):
        fh.return_data()
        split = model_params.get("training_split")
        data_dir = os.listdir(paths['origin_data'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        init_data_length = len([file for file in data_dir if os.path.isfile(os.path.join(paths['origin_data'], file))])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertEqual(init_training_length, 0)
        self.assertEqual(init_validation_length, 0)

        fh.split_data()

        data_dir = os.listdir(paths['origin_data'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(paths['origin_data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertEqual(data_length, 0)
        self.assertEqual(training_length, round(init_data_length*split/100))
        self.assertEqual(validation_length, round(init_data_length*(100-split)/100))
    

    def test_givenData_whenReturnData_thenDataIsReturned(self):
        fh.split_data()
        data_dir = os.listdir(paths['origin_data'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertNotEqual(init_training_length, 0)
        self.assertNotEqual(init_validation_length, 0)

        fh.return_data()

        data_dir = os.listdir(paths['origin_data'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(paths['origin_data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertEqual(data_length, init_training_length+init_validation_length)
        self.assertEqual(training_length, 0)
        self.assertEqual(validation_length, 0)
    
    def test_givenData_whenReturnPhaseData_thenPhaseDataIsReturned(self):
        fh.split_data()
        data_dir = os.listdir(paths['phase'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertNotEqual(init_training_length, 0)
        self.assertNotEqual(init_validation_length, 0)

        fh.return_data()

        data_dir = os.listdir(paths['origin_data'])
        training_dir = os.listdir(paths['training_data'])
        validation_dir = os.listdir(paths['validation_data'])
        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(paths['origin_data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(paths['training_data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(paths['validation_data'], file))])

        self.assertEqual(data_length, init_training_length+init_validation_length)
        self.assertEqual(training_length, 0)
        self.assertEqual(validation_length, 0)


if __name__ == '__main__':
    unittest.main()