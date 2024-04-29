import os
import random
data_path ="Data"
training_path = "Training"
validation_path = "Validation"
validation_amount = 1

def split_data():
    folders = [os.listdir(data_path)] #should be ["Data\Images", "Data\Masks"]
    images = [os.listdir(folders[0])]
    masks = [os.listdir(folders[1])]
    data = zip(images, masks)
    random.shuffle(data)
    #Training Data
    for image, mask in data[validation_amount:]:
        os.rename(os.path.join(data_path,"Images", image), os.path.join(training_path, "Images", image))
        os.rename(os.path.join(data_path,"Masks", mask), os.path.join(training_path, "Masks", mask))
    #Validation data
    for image, mask in data[:validation_amount]:
        os.rename(os.path.join(data_path,"Images", image), os.path.join(validation_path, "Images", image))
        os.rename(os.path.join(data_path,"Masks", mask), os.path.join(validation_path, "Masks", mask))

#===========================================================
def main():
    split_data()

if __name__ == "__main__":
    main()