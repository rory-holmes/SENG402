import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
modelPath = "0.torch"  # Path to trained model
imagePath = "Validation\Images\img_01_raw.png"  # Test image
width = 640
height = 480

transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor(),tf.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])  

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  
Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1)) 
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode
Img = cv2.imread(imagePath) # load test image
height_orgin , widh_orgin ,d = Img.shape # Get image original size 
plt.imshow(Img[:,:,::-1])  # Show image
plt.show()
Img = transformImg(Img)  # Transform to pytorch
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    Prd = Net(Img)['out']  # Run net
# resize to orginal size
Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0])#Convert probability to class map
seg = torch.argmax(Prd, 0).cpu().detach().numpy()  
plt.imshow(seg)  # display image
plt.show()