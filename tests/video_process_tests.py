import unittest
import os
import sys
sys.path.append('utils')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.video_process as vp
import utils.file_helpers as fh
import numpy as np
import yaml
import cv2

with open(r"params\\feature_model_params.yaml", "r") as f:
    model_params = yaml.load(f, Loader=yaml.SafeLoader)

with open(r"params\\phase_model_params.yaml", "r") as f:
    phase_model_params = yaml.load(f, Loader=yaml.SafeLoader)

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

def createVideoToTestPhaseSteps():
    """
    Creates a test_suite dir and adds a fake Video to test the amount of steps
    """
    os.makedirs('test_suite', exist_ok=True)
    filename = os.path.join('test_suite', 'testVideo.mpg')
    fourcc = cv2.VideoWriter_fourcc(*'MPG1')
    out = cv2.VideoWriter(filename, fourcc, 25, (250, 250))

    for i in range(50):
        frame = np.full((250, 250, 3), (i % 256, (i * 2) % 256, (i * 3) % 256), dtype=np.uint8)
        out.write(frame)
    out.release()

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
    
    def test_givenIncorrectPath_whenPhaseDataGenerator_thenValueError(self):
        with self.assertRaises(ValueError):
            _, _ = vp.phase_generator('Incorrect/Path')
    
    def test_givenCorrectPath_whenPhaseDataGenerator_thenBatchReturned(self):
        fh.split_phase_data()
        batch_size = phase_model_params['batch_size']
        for frames, labels in vp.phase_generator('training'):
            self.assertEqual(len(frames), len(labels))
            self.assertEqual(len(frames), batch_size)
            break
        fh.return_data(phase=True)

    def test_givenFrame_whenResizeFrame_thenFrameResized(self):
        frame = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        resized_frame = vp.resize_frame(frame)
        self.assertEqual(resized_frame.shape, ((n_h, n_w, 3)))

    def test_givenLabels_whenGetSteps_thenCorrectStepsReturned(self):
        createFolderToTestSteps()
        steps = vp.get_steps('test_suite')
        self.assertEqual(steps, 30/model_params['batch_size'])
        tearDownTestSuite()

    def test_givenFakeVideo_whenGetPhaseSteps_thenCorrectStepsReturned(self):
        createVideoToTestPhaseSteps()
        steps = vp.get_steps('test_suite')
        self.assertEqual(steps, 50)
        tearDownTestSuite()

if __name__ == '__main__':
    unittest.main()