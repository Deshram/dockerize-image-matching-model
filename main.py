import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import kornia as K
import kornia.feature as KF 
from kornia_moons.viz import draw_LAF_matches
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

#setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initializing api insance
app = FastAPI()

#Load model
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("./models/loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()

#bytes-image to tensor
def get_tensor_image(img_bytes):
    img = np.asarray(bytearray(img_bytes), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    scale = 840 / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

@app.post("/get_matching")
async def get_matching(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = get_tensor_image(await image1.read())
    img2 = get_tensor_image(await image2.read())

    input_dict = {"image0": K.color.rgb_to_grayscale(img1), 
              "image1": K.color.rgb_to_grayscale(img2)}

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999999, 10000)
    inliers = inliers > 0   

    fig,ax = draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                   'tentative_color': None, 
                   'feature_color': (0.2, 0.5, 1), 'vertical': False}, return_fig_ax = True)

    ax.axis('off')
    plt.savefig('output.jpg', bbox_inches='tight')
    return FileResponse("./output.jpg", media_type="image/jpeg")