import os
path = r"Validation"
upto_raw = 121
upto_class = 121
for p in os.listdir(path):
    path2 = os.path.join(path, p)
    for p2 in os.listdir(path2):
        print(p2)
        if "class" in p2:
            p2 = os.rename(os.path.join(path2, p2), os.path.join("Data\Masks",("img_"+ str(upto_class) + "_class.png")))
            upto_class +=1 
        elif "raw" in p2:
            p2 = os.rename(os.path.join(path2, p2), os.path.join("Data\Images",("img_"+ str(upto_raw) + "_raw.png")))
            upto_raw += 1
        