# SENG402
Operative skill assesment for lacroscopic colorectal surgery

## Table of Contents
[Description](#description)

[Installation](#installation)

[Setup](#Setup)

[Usage](#Usage)

[Future Steps](#Future-steps)

## Description
Although operative reviews have shown to be valuable in competency-based surgical training, manual reviews are too time-consuming to be a routine practice. Colorectal surgery has shown potential for skill assessment due to the correlation between Vascular Pedicle Dissection Time (VPDT) and manually assessed competency scores. This paper proposes a CNN that will be used to extract relevant features from laparoscopic videos combined with a 3D-CNN to quantify the VPDT and used to calculate an automatically produced competency score. The initial outcome of this approach is a feature extractor with 81% accuracy and an F1 score of 76%. While the phase detector underperformed with an accuracy of 29% and an F1 score of
3.9%.

See the documentation directory for full explanation of problem and implementation.

### Project Structure:
```
SENG402
├── .venv/                   # Virtual environment
├── documentation/           # Project documentation
│   ├── seng402_final_Holmes-Rory.pdf
│   └── Showcase_Holmes-Rory.pptx
├── models/                  # Model scripts
│   ├── CNN.py
│   └── PhaseCNN.py
├── params/                  # Parameter and hyperparameter configuration files
│   ├── feature_model_params.yaml
│   ├── params.yaml
│   └── phase_model_params.yaml
├── results/                 # Storage for model results and logs
├── tests/                   # Unit tests
│   ├── file_helpers_test.py
│   ├── global_helpers_test.py
│   └── video_process_tests.py
├── utils/                   # Utility scripts
│   ├── file_helpers.py
│   ├── global_helpers.py
│   └── video_process.py
├── .gitignore               # Git ignore file
├── pipeline.py              # Main pipeline script
├── README.md                # You are here 
└── requirements.txt         # Package dependencies
```

## Installation:

 1. Clone the repo:
    ```sh
    git clone https://github.com/rory-holmes/SENG402.git
    ```

 2. Naviagate to project directory:
    ```sh
    cd SENG402
    ```

 3. Install the [cholec80 dataset](http://camma.u-strasbg.fr/datasets). 

## Setup:
 1. Create a Virtual Environment to ensure dependencies are managed correctly:
   ```bash
   python3 -m venv venv
   ```

 2. Activate the environment:
   ```bash
   venv\Scripts\activate
   ```

 3. Install the depencencies
   ```bash
   pip install -r requirements.txt
   ```

 4. Run the ```setup()``` method found within ```file_helpers.py``` to initialise folder structure.

 5. Place cholec80 videos and annotations within the ```data/feature_extraction``` directory

 6. Place the phase detection videos and annotations within their respective paths in the ```data/phase_detection``` directory.

## Usage:

 1. Ensure pathways within ```params.yaml``` are correct.

 2. Optionally split some data into testing folder manually for testing.

 3. Call the appropriate methods within pipeline.py
   ```train_feature_extractor```, ```train_phase_detector```, ```full_training_cycle```
   Models will be saved in the results directory.

 4. Evaluate data via methods in the ```global_helpers.py``` file

## Future-steps:

### Feature Extraction:
- Gradually reduce learning rate with optimizers.schedules
- During cholec80 remove all frames where tools do not corolate well with colorectal surgery, potentially remove frames where its not in the body?
- Save checkpoint with [callback](https://www.tensorflow.org/tutorials/keras/save_and_load) usage.
- Acquire tool annotations for colorectal dataset.

### Phase Detection:
- Balance the dataset for each of the phases (crop the vidoes or augment).
- Change the classification to include phases before and after the VPD to help balance the dataset.
