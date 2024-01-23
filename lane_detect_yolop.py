import torch
from torchvision import transforms
import numpy as np
import cv2
import queue
import time
import os



class lane_detect_yolop():
    def __init__(self) -> None:
        self.model = None

        self.to_tensor_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize((640,640), antialias=True)
        self.normalize = transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([self.to_tensor_transform, self.resize_transform, self.normalize])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.start_time = None
        self.end_time = None
        self.time_q = queue.deque(maxlen=5)


    def load_model(self):
        self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True).to(self.device)

    def preprocess(self, input):
        features = self.transform(input).to(self.device)

        if features.ndimension() == 3:
            features = features.unsqueeze(0)

        return features

    def predict(self, image, en_op = False):
        
        if self.model is None:
            self.load_model()

        self.start_time = time.perf_counter()  # start counter

        features = self.preprocess(image)
    
        det_out, driveable_area_seg_out, lane_line_seg_out = self.model(features)
        _, lane_line_seg_out = torch.max(lane_line_seg_out, 1)
        _, driveable_area_seg_out = torch.max(driveable_area_seg_out, 1)

        ll_seg_mask = lane_line_seg_out.int().squeeze().cpu().numpy()
        da_seg_mask = driveable_area_seg_out.int().squeeze().cpu().numpy()

        output = self.show_seg_result(image, (da_seg_mask, ll_seg_mask))
        
        self.end_time = time.perf_counter()  # end counter

        self.time_q.append(((self.end_time - self.start_time)*1000))
       
        if en_op:

            if not len(self.time_q):
                print(f'avg exe_time: {round((self.end_time - self.start_time)*1000, 2)} ms')

            else:
                print(f'avg exe_time: {round(sum(self.time_q)/len(self.time_q), 2)} ms')


        return output
        
    # function taken from https://github.com/hustvl/YOLOP/blob/main/tools/demo.py
    def show_seg_result(self, img, result): 
        image_h, image_w = img.shape[1], img.shape[0] 

        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)

        # color_area[result[0] == 1] = [0, 255, 0]   #uncomment to display driveable area mask
        color_area[result[1] ==1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        # color_seg = color_seg[..., ::-1]
        color_seg = cv2.cvtColor(color_seg, cv2.COLOR_RGB2BGR)

        # print(color_seg.shape)
        color_mask = np.mean(color_seg, 2)
        img = cv2.resize(img, (640,640), interpolation=cv2.INTER_LINEAR)
        img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img = img.astype(np.uint8)
        img = cv2.resize(img, (image_h, image_w), interpolation=cv2.INTER_LINEAR)

        return img
    


def main():
    lane_detector = lane_detect_yolop()

    image_folder = "_out/rgb"  # replace with path to image dir
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(".png")]

    
    
    for image_path in image_paths:

        start_time = time.perf_counter()
        
        img = cv2.imread(image_path)

        op = lane_detector.predict(img)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time

        # Display FPS
        cv2.putText(op, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('output', op)
        cv2.waitKey(1)
        
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


