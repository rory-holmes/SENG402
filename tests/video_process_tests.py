import unittest
import os
import sys
sys.path.append('utils')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.video_process as vp
import utils.file_helpers as fh
import numpy as np
import yaml

with open("params/feature_model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

n_w = model_params.get("image_width")
n_h = model_params.get("image_height")

def createFolderToTestSteps():
    """
    Creates a directory with 2 files in each with 15 lines for step testing
    """
    os.makedirs('test_suite', exist_ok=True)

    file1_name = os.path.join('test_suite', 'file1.txt')
    file2_name = os.path.join('test_suite', 'file2.txt')
    content = [f"This is line {i+1}\n" for i in range(15)]

    with open(file1_name, 'w') as file1:
        file1.writelines(content)
    with open(file2_name, 'w') as file2:
        file2.writelines(content)

def tearDownTestSuite():
    """
    Delete all files within and the 'test_suite' dir
    """
    if os.path.exists('test_suite'):
        for filename in os.listdir('test_suite'):
            file_path = os.path.join('test_suite', filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir('test_suite')

class TestStringMethods(unittest.TestCase):

    def test_givenIncorrectPath_whenDataGenerator_thenValueError(self):
        with self.assertRaises(ValueError):
            _, _ = vp.data_generator('Incorrect/Path', 0)

    def test_givenCorrectPath_whenDataGenerator_thenBatchReturned(self):
        fh.split_data()
        batch_size = 5
        for frames, labels in vp.data_generator('training', batch_size):
            self.assertEqual(len(frames), len(labels))
            self.assertEqual(len(frames), batch_size)
            break
        fh.return_data()

    def test_givenFrame_whenResizeFrame_thenFrameResized(self):
        frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        resized_frame = vp.resize_frame(frame)
        self.assertEqual(resized_frame.shape, ((n_h, n_w, 3)))

    def test_givenLabels_whenGetSteps_thenCorrectStepsReturned(self):
        createFolderToTestSteps()
        steps = vp.get_steps('test_suite')
        self.assertEqual(steps, 30/model_params['batch_size'])
        tearDownTestSuite()

if __name__ == '__main__':
    unittest.main()