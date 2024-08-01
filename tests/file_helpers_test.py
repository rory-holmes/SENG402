import unittest
import os
import sys
sys.path.append('params')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.file_helpers as fh
import yaml

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

class TestSringMethods(unittest.TestCase):
    def test_givenData_whenSplitData_thenDataSplitCorrectly(self):
        fh.return_data()
        split = model_params.get("training_split")
        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])
        init_data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        self.assertEqual(init_training_length, 0)
        self.assertEqual(init_validation_length, 0)

        fh.split_data()

        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])
        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        self.assertEqual(data_length, 0)
        self.assertEqual(training_length, round(init_data_length*split/100))
        self.assertEqual(validation_length, round(init_data_length*(100-split)/100))
    

    def test_givenData_whenReturnData_thenDataIsReturned(self):
        fh.split_data()
        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        self.assertNotEqual(init_training_length, 0)
        self.assertNotEqual(init_validation_length, 0)

        fh.return_data()

        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])
        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        self.assertEqual(data_length, init_training_length+init_validation_length)
        self.assertEqual(training_length, 0)
        self.assertEqual(validation_length, 0)



if __name__ == '__main__':
    unittest.main()