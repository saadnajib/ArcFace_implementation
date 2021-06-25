import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


# parser = argparse.ArgumentParser(description='for face verification')
# parser.add_argument("-s", "--save", help="whether save",action="store_true")
# parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
# parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
# parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
# parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
# args = parser.parse_args()

def rescale_video_img(frame1,scale=0.75):
    width = int(frame1.shape[1] * scale)
    height = int(frame1.shape[0] * scale)
    dim = (width,height)
    return (cv2.resize(frame1,dim,interpolation=cv2.INTER_AREA))

def main(args):

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

    targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
    print('facebank updated')



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
    args.video_speedup = 2
    main(args)         
    
    

   


