import os
import glob
import cv2
import torch
import numpy as np

from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator

index = '999'

height = 256
width = 256

angle = 0.0 # +:/→|    -:\→|
scale = 1.0

load_path = './dataset/'+index+'/original/'
data_path = './dataset/'+index+'/image/'
mask_path = './dataset/'+index+'/mask/'
edge_path = './dataset/'+index+'/edge/'
sket_path = './dataset/'+index+'/sketch/'

det_config='../mmdetection/datas/FaceDetect/det_config/config.py'
det_checkpoint='../mmdetection/datas/FaceDetect/det_checkpoint/300epoch_100img.pth'
pose_config = '../mmpose/datas/kp42/config/custom_config_kp42.py'
pose_checkpoint = '../mmpose/datas/kp42/checkpoint/epoch_60.pth'

device = 'cuda:0'

detector = init_detector(det_config, det_checkpoint, device=device)
pose_estimator = init_pose_estimator(pose_config, pose_checkpoint, device=device, cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True))))

def trimming():
    def estimate_bbox(img):
        init_default_scope(detector.cfg.get('default_scope', 'mmdet'))
        detect_result = inference_detector(detector, img)
        pred_instance = detect_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.1)]

        return bboxes
    
    def rotate():
        index = 0

        for img_path in file_list:
            filename = str(index)+'.jpg'

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            height = img.shape[0]                         
            width = img.shape[1]                         
            center = (int(width/2), int(height/2))

            trans = cv2.getRotationMatrix2D(center, angle , scale)
            rotated_img = cv2.warpAffine(img, trans, (width,height))

            cv2.imwrite(os.path.join(data_path, filename), rotated_img)

            index += 1

    def resize_bbox(x1, y1, x2, y2, scale):
        centerx = int((x2+x1)/2)
        centery = int((y2+y1)/2)

        newx1 = int(centerx-(centerx-x1)*scale)
        newy1 = int(centery-(centery-y1)*scale)
        newx2 = int(centerx+(x2-centerx)*scale)
        newy2 = int(centery+(y2-centery)*scale)

        return newx1, newy1, newx2, newy2

    def trim_bbox(bbox):
        index = 0

        x1 = int(bbox[0][0])
        y1 = int(bbox[0][1])
        x2 = int(bbox[0][2])
        y2 = int(bbox[0][3])

        x1, y1, x2, y2 = resize_bbox(x1, y1, x2, y2, 1.1)

        for img_path in file_list:
            filename = str(index)+'.jpg'

            img  = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_UNCHANGED)

            cropped_img = img[y1:y2, x1:x2]

            cv2.imwrite(os.path.join(data_path, filename), cropped_img)

            index += 1

        return (x2-x1), (y2-y1)

    file_list = glob.glob(os.path.join(load_path, "*.png"))
    top_img = os.path.join(data_path, "0.jpg")

    rotate()
    bbox = estimate_bbox(top_img)
    x, y = trim_bbox(bbox)

    return x, y

def makemask(x, y):
    def estimate_keypoints(img):
        imagebox = [[0, 0, x, y]]

        pose_results = inference_topdown(pose_estimator, img, imagebox)

        keypoints = pose_results[0].pred_instances.keypoints

        result = np.squeeze(keypoints)

        return result
    
    def drawmask():
        index = 0

        for img_path in file_list:
            filename = str(index)+'.jpg'

            img = os.path.join(data_path, filename)

            keypoints = estimate_keypoints(img)

            mask = np.full((y, x, 3), 0, dtype=np.uint8)

            pts = np.array(((int(keypoints[0][0]), int(keypoints[0][1])),
                            (int(keypoints[1][0]), int(keypoints[1][1])),
                            (int(keypoints[2][0]), int(keypoints[2][1])),
                            (int(keypoints[3][0]), int(keypoints[3][1])),
                            (int(keypoints[4][0]), int(keypoints[4][1]))))
            cv2.polylines(mask, [pts], False, (255, 255, 255), thickness=10, lineType=cv2.LINE_8)

            pts = np.array(((int(keypoints[5][0]), int(keypoints[5][1])),
                            (int(keypoints[6][0]), int(keypoints[6][1])),
                            (int(keypoints[7][0]), int(keypoints[7][1])),
                            (int(keypoints[8][0]), int(keypoints[8][1])),
                            (int(keypoints[9][0]), int(keypoints[9][1]))))
            cv2.polylines(mask, [pts], False, (255, 255, 255), thickness=10, lineType=cv2.LINE_8)

            pts = np.array(((int(keypoints[19][0]),  int(keypoints[19][1])),
                            (int(keypoints[10][0]),  int(keypoints[10][1])),
                            (int(keypoints[11][0]),  int(keypoints[11][1])),
                            (int(keypoints[12][0]),  int(keypoints[12][1])),
                            (int(keypoints[13][0]),  int(keypoints[13][1])),
                            (int(keypoints[14][0]),  int(keypoints[14][1]))))
            cv2.polylines(mask, [pts], False, (255, 255, 255), thickness=15, lineType=cv2.LINE_8)

            pts = np.array(((int(keypoints[10][0]),  int(keypoints[10][1])),
                            (int(keypoints[11][0]),  int(keypoints[11][1])),
                            (int(keypoints[12][0]),  int(keypoints[12][1])),
                            (int(keypoints[13][0]),  int(keypoints[13][1])),
                            (int(keypoints[14][0]),  int(keypoints[14][1])),
                            (int(keypoints[15][0]),  int(keypoints[15][1])),
                            (int(keypoints[16][0]),  int(keypoints[16][1])),
                            (int(keypoints[17][0]),  int(keypoints[17][1])),
                            (int(keypoints[18][0]),  int(keypoints[18][1])),
                            (int(keypoints[19][0]),  int(keypoints[19][1]))))
            cv2.polylines(mask, [pts], True, (255, 255, 255), thickness=10, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))

            pts = np.array(((int(keypoints[29][0]),  int(keypoints[29][1])),
                            (int(keypoints[20][0]),  int(keypoints[20][1])),
                            (int(keypoints[21][0]),  int(keypoints[21][1])),
                            (int(keypoints[22][0]),  int(keypoints[22][1])),
                            (int(keypoints[23][0]),  int(keypoints[23][1])),
                            (int(keypoints[24][0]),  int(keypoints[24][1]))))
            cv2.polylines(mask, [pts], False, (255, 255, 255), thickness=15, lineType=cv2.LINE_8)

            pts = np.array(((int(keypoints[20][0]),  int(keypoints[20][1])),
                            (int(keypoints[21][0]),  int(keypoints[21][1])),
                            (int(keypoints[22][0]),  int(keypoints[22][1])),
                            (int(keypoints[23][0]),  int(keypoints[23][1])),
                            (int(keypoints[24][0]),  int(keypoints[24][1])),
                            (int(keypoints[25][0]),  int(keypoints[25][1])),
                            (int(keypoints[26][0]),  int(keypoints[26][1])),
                            (int(keypoints[27][0]),  int(keypoints[27][1])),
                            (int(keypoints[28][0]),  int(keypoints[28][1])),
                            (int(keypoints[29][0]),  int(keypoints[29][1]))))
            cv2.polylines(mask, [pts], True, (255, 255, 255), thickness=10, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))

            pts = np.array(((int(keypoints[41][0]), int(keypoints[41][1])),
                            (int(keypoints[32][0]), int(keypoints[32][1])),
                            (int(keypoints[33][0]), int(keypoints[33][1])),
                            (int(keypoints[34][0]), int(keypoints[34][1])),
                            (int(keypoints[35][0]), int(keypoints[35][1])),
                            (int(keypoints[36][0]), int(keypoints[36][1])),
                            (int(keypoints[37][0]), int(keypoints[37][1])),
                            (int(keypoints[38][0]), int(keypoints[38][1])),
                            (int(keypoints[39][0]), int(keypoints[39][1])),
                            (int(keypoints[40][0]), int(keypoints[40][1]))))
            cv2.polylines(mask, [pts], True, (255, 255, 255), thickness=15, lineType=cv2.LINE_8)
            cv2.fillPoly(mask, [pts], (255, 255, 255))

            cv2.imwrite(os.path.join(mask_path, filename), mask)

            index += 1

    file_list = glob.glob(os.path.join(data_path, "*.jpg"))

    drawmask()

def resize():
    file_list = glob.glob(os.path.join(data_path, "*.jpg"))

    index = 0

    for img_path in file_list:
        filename = str(index)+'.jpg'

        img  = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_UNCHANGED)

        reimg  = cv2.resize(img,  (height, width))
        remask = cv2.resize(mask, (height, width))

        cv2.imwrite(os.path.join(data_path, filename), reimg)
        cv2.imwrite(os.path.join(mask_path, filename), remask)

        index += 1

def makeedge():
    file_list = glob.glob(os.path.join(data_path, "*.jpg"))

    index = 0

    for img_path in file_list:
        filename = str(index)+'.jpg'

        img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_UNCHANGED)

        edge = cv2.Canny(img, 40, 80, L2gradient=True) #(40, 80) or (80, 150)

        cv2.imwrite(os.path.join(edge_path, filename), edge)

        index += 1

def makesketch():
    file_list = glob.glob(os.path.join(edge_path, "*.jpg"))

    index = 0

    for img_path in file_list:
        filename = str(index)+'.jpg'

        edge = cv2.imread(os.path.join(edge_path, filename))
        mask = cv2.imread(os.path.join(mask_path, filename))

        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        edge = torch.from_numpy(edge.astype(np.float32) / 255.0).view(1, 256, 256).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).view(1, 256, 256).contiguous()

        sketch = edge * mask
        sketch = sketch * 255

        sketch_copy = sketch.clone().data.permute(1, 2, 0).numpy()
        sketch_copy = np.clip(sketch_copy, 0, 255)
        sketch_copy = sketch_copy.astype(np.uint8)
        sketch_copy = cv2.cvtColor(sketch_copy, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(sket_path, filename), sketch_copy)

        index += 1

x, y = trimming()
makemask(x,y)
resize()
makeedge()
makesketch()