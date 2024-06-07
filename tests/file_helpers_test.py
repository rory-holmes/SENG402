import unittest
import os
import sys
sys.path.append('utils')
from ..utils import file_helpers as fh
import yaml

with open("params\params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.SafeLoader)

with open("params\model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

class TestValueError(unittest.TestCase):
    def givenData_whenSplitData_thenDataSplitCorrectly():
        fh.return_data()
        split = model_params.get("training_split")
        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])

        init_data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        assert(init_training_length == 0)
        assert(init_validation_length == 0)

        fh.split_data()

        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        assert(data_length == 0)
        assert(training_length == (init_data_length*split/100))
        assert(validation_length == (init_data_length*(100-split)/100))
    

    def givenData_whenReturnData_thenDataIsReturned():
        fh.split_data()
        data_dir = os.listdir(params['origin_path']['data'])
        training_dir = os.listdir(params['training_path']['data'])
        validation_dir = os.listdir(params['validation_path']['data'])

        init_training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        init_validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        assert(init_training_length != 0)
        assert(init_validation_length != 0)

        fh.return_data()

        data_length = len([file for file in data_dir if os.path.isfile(os.path.join(params['origin_path']['data'], file))])
        training_length = len([file for file in training_dir if os.path.isfile(os.path.join(params['training_path']['data'], file))])
        validation_length = len([file for file in validation_dir if os.path.isfile(os.path.join(params['validation_path']['data'], file))])

        assert(data_length == init_training_length+init_validation_length)
        assert(training_length == 0)
        assert(validation_length == 0)



if __name__ == '__main__':
    unittest.main()