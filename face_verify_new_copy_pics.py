import cv2
from PIL import Image, ImageDraw, ImageFont
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import time
import torch.onnx
import onnx
import glob



def rescale_video_img(frame1,scale=0.75):
    width = int(frame1.shape[1] * scale)
    height = int(frame1.shape[0] * scale)
    dim = (width,height)
    return (cv2.resize(frame1,dim,interpolation=cv2.INTER_AREA))

def convertTuple(tup):
    str =  '_'.join(tup)
    return str

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

def main(args):
    same_count = 0
    not_same_count = 0

    conf = get_config(False)

    mtcnn = MTCNN()
    print('arcface loaded')

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()


    print('learner loaded')

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')


# full = /home/saad/Desktop/training_dataset/attendance/Ahmed_Nawaz_7_right_289_489_108_104_06102021_061847.jpg
# name_list_image = Abdul_Reheem_61_left_1527_564_84_89_06122021_060805.jpg
# str1 = Imran_Suleman_116__797_462_82_87_06142021_084243
# name_list = Imran_Suleman_116

    photos = glob.glob('/home/saad/Desktop/training_dataset/attendance/*')
    full_path = []
    name_list = []
    # unique_name_list = []
    # name_list_image = []
    for p in photos:
        full = p
        str1 = p.split('/')[-1].replace(' ', '').replace('right', '').replace('random', '').replace('left', '').replace('front', '').replace('.jpg', '')
        str2 = str1.split('_')[:3]
        str3 = convertTuple(str2)

        full_path.append(full)
        name_list.append(str3)
        # name_list_image.append(p.split('/')[-1]) 

    # print(name_list)

    for elem in enumerate(full_path):
        try:
            frame = cv2.imread(elem[1])
            # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            start_time = time.time()
            # frame = resize_image(frame, 0.5)
            image = Image.fromarray(frame)
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            results, score = learner.infer(conf, faces, targets, args.tta)
            # print(score[0])

            FPS = 1.0 / (time.time() - start_time)

            frame = cv2.putText(frame,
                    'FPS: {:.1f}'.format(FPS),(10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255,0,0),
                    2,
                    cv2.LINE_AA)

            for idx,bbox in enumerate(bboxes):
                if args.score:
                    frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                else:
                    if float('{:.2f}'.format(score[idx])) > .94:
                        # name = names[results[idx]+1]
                        name = names[0]
                        print("if score {:.3f}".format(score[idx]))
                        # print("if name %s" %name)
                    else:
                        # name = names[0]    
                        name = names[results[idx]+1]
                        print("else score {:.3f}".format(score[idx]))
                        # print("else name %s" %name)

            frame = draw_box_name(bbox, name, frame)

            if(name == name_list[elem[0]]):
                print("Same Name")
                same_count = same_count + 1
            else:
                not_same_count = not_same_count + 1
                print("Not Same")
            
            print("orignal name "+ name_list[elem[0]] )
            print("Predicted name "+ name )


        except:
            pass 

        # cv2.imshow('video', frame)
        # cv2.waitKey(0) 


    print("total correct recognize: ",same_count)
    print("total incorrect recognize: ",not_same_count)




if __name__ == "__main__":
    args = lambda : None
    args.width = 800
    args.height = 800
    args.image = None
    args.save = True
    args.threshold = 1.54
    args.update = False
    args.tta = True
    args.score = False
    args.video_speedup = False
    main(args)         
    
    

   


