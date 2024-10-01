# SENG402
Operative skill assesment for lacroscopic colorectal surgery

## Table of Contents
[Description](#description)

[Installation](#installation)

[Usage](#Usage)

## Description
This code focuses on assessing surgical skill within colorectal surgery. Although operative reviews have shown to be valuable in competency-based surgical training, manual reviews are too time-consuming to be a routine practice. Colorectal surgery has shown potential for skill assessment due to the correlation between Vascular Pedicle Dissection Time (VPDT) and manually assessed competency scores. This code focuses on the feature extraction of tool classification within cholocysectomy surgery to be used for phase detection in colorectal surgery.

## Installation:

 1. Clone the repo:
    ```sh
    git clone https://github.com/rory-holmes/SENG402.git
    ```
 2. Naviagate to project directory:
    ```sh
    cd SENG402
    ```
 4. Install project dependencies found with in dependencies.json
    ```sh
    npm install
    ```
 5. Install the [cholec80 dataset](http://camma.u-strasbg.fr/datasets). Or Endovis15*

## Usage:

 1. Run the setup method found within file_helpers to initialse folders for training.
 2. Setup pathways found within params.yaml
 3. Split Some data into testing folder manually for testing.
 4. Run pipeline.py
 5. Evaluate data by the global_helpers.py file
