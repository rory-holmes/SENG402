# SENG402
Operative skill assesment for lacroscopic colorectal surgery

Installation:

 1. Naviagate to project directory.

 2. Install project dependencies found with in dependencies.json
    
    npm install

 3. Install cholec80 dataset.

    http://camma.u-strasbg.fr/datasets

Usage:

 1. Run the setup method found within file_helpers to initialse folders for training.
 2. Setup pathways found within params.yaml
 3. Run pipeline.py
 4. Test output of model by calling the demo() method within global_helpers.py with the path to the trained model.