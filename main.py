import cv2
import torch
import torchvision.models.segmentation
import torchvision.transforms as tf
import numpy as np
from PIL import Image
import os

training_folder="Training"
training_images=os.listdir("Training\Images")
training_masks=os.listdir("Training\Masks")
validation_images=os.listdir("Validation\Images")
validation_masks=os.listdir("Validation\Masks")
width = 640
height = 480
batchSize=4
#Step size of gradient descent
Learning_Rate=1e-5

transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)), tf.ToTensor()])
      
def ReadRandomImage():
    """ Selects random image from the training folder, returns it and its map"""   
    idx = np.random.randint(0,len(training_images)) # Pick random image   
    Img = cv2.imread(os.path.join(training_folder, "Images", training_images[idx]))  
    class_map = cv2.imread(os.path.join(training_folder, "Masks", training_images[idx][0:6] + "_class.png"))
    AnnMap = np.zeros(Img.shape[0:2],np.float32) # Segmentation map 
    if class_map is not None: 
        for idx, i in enumerate(AnnMap):
            for idx2, j in enumerate(i):
                if np.array_equal(class_map[idx][idx2], np.array([160, 160, 160])):
                    AnnMap[idx][idx2] = 1
                if np.array_equal(class_map[idx][idx2], np.array([70,70,70])):
                    AnnMap[idx][idx2] = 2

        #AnnMap[ class_map == np.array("(160 160 160)") ] = 1    
        #AnnMap[ class_map == np.array("(70 70 70)") ] = 2  
    Img=transformImg(Img)
    AnnMap=transformAnn(AnnMap)  
    return Img,AnnMap

def LoadBatch():
    """Loads a batch of images"""
    images = torch.zeros([batchSize, 3, height, width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i], ann[i] = ReadRandomImage()

    return images, ann


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) 
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)

for itr in range(20000):
    images, ann = LoadBatch()
    images = torch.autograd.Variable(images, requires_grad=False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    Pred=Net(images)['out']
    criterion = torch.nn.CrossEntropyLoss()
    Loss = criterion(Pred, ann.long())
    Loss.backward()
    optimizer.step()
    if itr % 1000 == 0: 
        print("Saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(), str(itr) + ".torch")
        print("Saved model", str(itr))