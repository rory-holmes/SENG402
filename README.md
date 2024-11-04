# SENG402
Operative skill assesment for lacroscopic colorectal surgery

## Table of Contents
[Description](#description)

[Installation](#installation)

[Setup](#Setup)

[Usage](#Usage)

## Description
Although operative reviews have shown to be valuable in competency-based surgical training, manual reviews are too time-consuming to be a routine practice. Colorectal surgery has shown potential for skill assessment due to the correlation between Vascular Pedicle Dissection Time (VPDT) and manually assessed competency scores. This paper proposes a CNN that will be used to extract relevant features from laparoscopic videos combined with a 3D-CNN to quantify the VPDT and used to calculate an automatically produced competency score. The initial outcome of this approach is a feature extractor with 81% accuracy and an F1 score of 76%. While the phase detector underperformed with an accuracy of 29% and an F1 score of
3.9%.

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

 4. Run the setup method found within file_helpers to initialise folder structure.

 5. Place cholec80 videos and annotations within the ```data/feature_extraction``` directory

 6. Place the phase detection videos and annotations within their respective paths in the ```data/phase_detection``` directory.

## Usage:

 1. Ensure pathways within params.yaml are correct.

 2. Optionally split some data into testing folder manually for testing.

 3. Call the appropriate methods within pipeline.py
   ```train_feature_extractor```, ```train_phase_detector```, ```full_training_cycle```
   Models will be saved in the results directory.

 4. Evaluate data via methods in the global_helpers.py file
