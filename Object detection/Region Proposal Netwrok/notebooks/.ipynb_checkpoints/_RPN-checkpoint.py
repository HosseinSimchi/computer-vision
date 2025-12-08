#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchsummary import summary
import io
import sys
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork, AnchorGenerator
import shap

# In[2]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[3]:


DEVICE

# In[4]:


IMAGE_DIR = '../dataset/images'
ANNOTATIONS_DIR = '../dataset/annotations'
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8

# In[5]:


class_names = [d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))]

# In[6]:


class_names

# In[7]:


csv_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.csv')]

# In[8]:


label_map = {name: i + 1 for i, name in enumerate(class_names)}

# In[9]:


label_map

# In[10]:


dataset = []

for i in range(len(class_names)):
    class_name = class_names[i]
    class_dir = os.path.join(IMAGE_DIR, class_name)
    csv_file_name = csv_files[i]
    
    csv_path = os.path.join(ANNOTATIONS_DIR, csv_file_name)
    df_annotations = pd.read_csv(csv_path)

    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        image = cv2.imread(image_path)
        if image is None:
            continue
                
        h, w, _ = image.shape

        row = df_annotations[df_annotations['image_name'] == image_name]

        # ---- FIX HERE ----
        if row.empty:
            continue
        # -------------------

        ann = row.iloc[0, 1:].tolist()

        ann[0] = (ann[0] / w) * 224
        ann[1] = (ann[1] / h) * 224
        ann[2] = (ann[2] / w) * 224
        ann[3] = (ann[3] / h) * 224

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label_tensor = torch.tensor([class_names.index(class_name)], dtype=torch.int64)
        ann_tensor = torch.tensor([ann], dtype=torch.float32)

        target = {
            'boxes': ann_tensor,
            'labels': label_tensor
        }
        dataset.append((image_tensor, target))

def collate_fn(batch):
    return tuple(zip(*batch))

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


# In[11]:


resnet_model = torchvision.models.resnet18()

# In[ ]:


summary(resnet_model,(3,224,224))

# In[12]:


backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
backbone.out_channels = 512 # custom attribute
backbone.out_channels

# In[13]:


output_dir = "../model_outputs"
os.makedirs(output_dir, exist_ok=True)

# file path
save_path = os.path.join(output_dir, "model_summary.txt")

# capture printed summary
buffer = io.StringIO()
sys_stdout = sys.stdout  # save real stdout
sys.stdout = buffer      # redirect stdout

summary(backbone, (3, 224, 224))   # <-- prints inside buffer
summary(backbone,(3,224,224))

sys.stdout = sys_stdout  # restore real stdout

# write to file
with open(save_path, "w") as f:
    f.write(buffer.getvalue())

print("Model summary saved to:", save_path)

# In[14]:


for param in backbone.parameters():
    param.requires_grad = False

# In[15]:


anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128),),
    aspect_ratios=((0.5, 1.0, 2.0),)
)

# In[16]:


in_channels = backbone.out_channels
num_anchors = anchor_generator.num_anchors_per_location()[0]

rpn_head = RPNHead(in_channels=in_channels, num_anchors=num_anchors)

# In[17]:


rpn_model = RegionProposalNetwork(
    anchor_generator,
    rpn_head,
    fg_iou_thresh=0.7,
    bg_iou_thresh=0.3,
    batch_size_per_image=256,
    positive_fraction=0.5,
    pre_nms_top_n={'training': 2000, 'testing': 1000},
    post_nms_top_n={'training': 1000, 'testing': 500},
    nms_thresh=0.7
)

# In[19]:


optimizer = torch.optim.Adam(rpn_model.parameters(), lr=0.001)

# In[20]:


import os

log_dir = "../model_outputs"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "training_log.txt")

NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    
    for images, targets in dataloader:
        optimizer.zero_grad()
        
        images_gpu = torch.stack([img.to(DEVICE) for img in images])
        targets_gpu = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            features = backbone(images_gpu)

        image_list = ImageList(images_gpu, [img.shape[-2:] for img in images])
        
        _, loss_dict = rpn_model(image_list, {'0': features}, targets_gpu)
        loss = loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg']
        
        if torch.isfinite(loss):
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
    if epoch_losses:
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        log_line = f"Epoch {epoch+1} | Loss: {mean_loss:.4f}\n"
    else:
        log_line = f"Epoch {epoch+1} | No valid losses\n"

    print(log_line.strip())

    # Save log
    with open(log_path, "a") as f:
        f.write(log_line)


# In[21]:


def visualize_rpn_proposals(image_path, rpn_model_trained, backbone_model_trained):
    import os
    from datetime import datetime

    # Output directory
    output_dir = "../model_outputs/rpn_proposals"
    os.makedirs(output_dir, exist_ok=True)

    TARGET_SIZE = (224,224)
    rpn_model_trained.to(DEVICE)
    backbone_model_trained.to(DEVICE)
    rpn_model_trained.eval()
    backbone_model_trained.eval()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    image_rgb_resized = cv2.resize(img_rgb, TARGET_SIZE)
    img_tensor = (torch.tensor(image_rgb_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0).to(DEVICE)

    with torch.no_grad():
        features = backbone_model_trained(img_tensor.unsqueeze(0))
        image_list = ImageList(img_tensor.unsqueeze(0), [tuple(img_tensor.shape[-2:])])
        proposals, _ = rpn_model_trained(image_list, {'0': features})

    top_proposals = proposals[0][:5].cpu().numpy()

    print(f"\nTotal proposals (after NMS): {len(proposals[0])}")
    print("Drawing top 5 proposals on the image...")

    # Work on a copy to avoid drawing on original resized image
    img_display = image_rgb_resized.copy()

    for i, box in enumerate(top_proposals):
        x1, y1, x2, y2 = map(int, box)

        if i == 0:
            color_rgb, width = (0, 255, 0), 3
        elif i < 3:
            color_rgb, width = (255, 255, 0), 2
        else:
            color_rgb, width = (255, 0, 0), 1

        cv2.rectangle(img_display, (x1, y1), (x2, y2), color_rgb, width)

    # ---- Save image ----
    base_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{base_name}_rpn_proposals.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(img_display)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved proposal visualization to: {output_path}")


# In[22]:


TEST_IMAGE_PATH = '../dataset/images/airplane/image_0002.jpg'
visualize_rpn_proposals(TEST_IMAGE_PATH, rpn_model, backbone)


# In[ ]:



