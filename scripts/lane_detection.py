#! /usr/bin/env python3

import rospy
import sys, time
import timeit
from sensor_msgs.msg import CompressedImage

import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image


args, cfg = merge_config()
img_w, img_h = 1280,720
row_anchor = tusimple_row_anchor
cls_num_per_lane = len(row_anchor)

net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4), use_aux=False).cuda() # we dont need auxiliary segmentation in testing

state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
camera_matrix = np.array([[7.666059408854959e+02 ,0 ,6.344911873102793e+02],[0 ,7.673519596803199e+02 ,3.603987305379446e+02],[0, 0, 1]])
dirtortion_matrix = np.array([0.196408429979919, -0.458374354770657, 0, 0, 0.255505894617166])
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,dirtortion_matrix,(1280,720),1,(1280,720))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix,dirtortion_matrix,None, new_camera_matrix, (1280,720), 5) 

class Lane:
    def __init__(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
        torch.backends.cudnn.benchmark = True
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",CompressedImage,self.cam_callback,queue_size=1)
        self.pub =  rospy.Publisher("/lane/image_raw/compressed",CompressedImage)

    def cam_callback(self, ros_data):
        start_t = timeit.default_timer()
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
        image_np = cv2.remap(image_np,mapx,mapy, cv2.INTER_LINEAR)



        imgs = img_transforms(Image.fromarray(image_np))
        imgs = imgs.unsqueeze(0)
        imgs = imgs.cuda()
        
        with torch.no_grad():
            out = net(imgs)

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]


        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc
        # import pdb; pdb.set_trace()
        
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(image_np,ppp,5,(0,255,0),-1)
        msg= CompressedImage()
        msg.header.stamp =rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        self.pub.publish(msg)
        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t ))
        cv2.waitKey(FPS)
        rospy.loginfo(f"activation\nFPS:{FPS}")

    
   
            

def main(args):
    lane =Lane()
    rospy.init_node("lane_detection",anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image Processing")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)



