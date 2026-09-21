import os, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, sobel

SEED=42
EPOCHS=8
BATCH_SIZE=32
LR=2e-4
WEIGHT_DECAY=1e-4
IMG_SIZE=224
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

def find_image(folder,image_id):
    for root,_,files in os.walk(folder):
        if image_id in files:
            return os.path.join(root,image_id)
    raise FileNotFoundError(image_id)

class LunarDataset(Dataset):
    def __init__(self,meta,folder):
        self.meta=meta.reset_index(drop=True)
        self.folder=folder

    def __len__(self):
        return len(self.meta)

    def __getitem__(self,i):
        r=self.meta.iloc[i]
        img=np.array(Image.open(find_image(self.folder,r.image_id)).convert("L"),dtype=np.float32)/255.0
        az=float(r.sun_azimuth_angle)

        img=rotate(img,-az,reshape=False,order=1,mode="reflect")

        blur=gaussian_filter(img,3)
        contrast=img-blur
        contrast=(contrast-contrast.min())/(contrast.max()-contrast.min()+1e-8)

        gx=sobel(img,axis=1)
        gy=sobel(img,axis=0)
        grad=np.sqrt(gx*gx+gy*gy)
        grad=(grad-grad.min())/(grad.max()-grad.min()+1e-8)

        def resize(a):
            return np.array(
                Image.fromarray(np.uint8(np.clip(a,0,1)*255)).resize((IMG_SIZE,IMG_SIZE)),
                dtype=np.float32
            )/255.0

        img,contrast,grad=resize(img),resize(contrast),resize(grad)
        rad=np.deg2rad(az)

        x=np.stack([
            img,
            contrast,
            grad,
            np.full_like(img,np.sin(rad)),
            np.full_like(img,np.cos(rad))
        ])

        return torch.tensor(x,dtype=torch.float32),int(r.label)

def build_model():
    m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    old=m.conv1
    new=nn.Conv2d(5,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=False)

    with torch.no_grad():
        new.weight[:,:3]=old.weight
        mean=old.weight.mean(dim=1,keepdim=True)
        new.weight[:,3:]=mean.repeat(1,2,1,1)

    m.conv1=new
    m.fc=nn.Sequential(nn.Dropout(0.35),nn.Linear(m.fc.in_features,2))
    return m

def main():
    meta=pd.read_csv("train_metadata.csv")
    ds=LunarDataset(meta,"train_images")
    loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

    model=build_model().to(DEVICE)

    counts=np.bincount(meta.label)
    weights=torch.tensor([len(meta)/(2*c) for c in counts],dtype=torch.float32,device=DEVICE)

    criterion=nn.CrossEntropyLoss(weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)

    for epoch in range(EPOCHS):
        model.train()
        correct=total=0

        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            optimizer.zero_grad()
            out=model(x)
            loss=criterion(out,y)
            loss.backward()
            optimizer.step()

            total+=len(y)
            correct+=(out.argmax(1)==y).sum().item()

        print(f"Epoch {epoch+1}/{EPOCHS} | accuracy={correct/total:.4f}")

    os.makedirs("models/revision7/final",exist_ok=True)
    path="models/revision7/final/revision7_full.pt"
    torch.save(model.state_dict(),path)
    print("Saved:",path)

if __name__=="__main__":
    main()
