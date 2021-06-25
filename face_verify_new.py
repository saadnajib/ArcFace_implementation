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

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap = cv2.VideoCapture(r"D:\face_recog\Face-Recognition_paul\videos\test2.mp4")
    cap.set(3,500)
    cap.set(4,500)    
    video_capture = cv2.VideoCapture(r"D:\face_recog\Face-Recognition_paul\videos\test2.mp4")
    widthh = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    heightt = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    _,frame11 = cap.read()
    frame11 = rescale_video_img(frame11,0.5)
    video_recording = cv2.VideoWriter(r'D:\face_recog\Face-Recognition_paul\output_video\output3.avi', fourcc, 10,(frame11.shape[1], frame11.shape[0]))
    
    total_frames_passed = 0

    while cap.isOpened():
        
        isSuccess,frame = cap.read()
        frame = rescale_video_img(frame,0.5)
        if args.video_speedup:               
            total_frames_passed += 1
            if total_frames_passed % args.video_speedup != 0:
                continue
    
        if isSuccess:            
            try:
                # image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets, args.tta)
                # print(score[0])
                for idx,bbox in enumerate(bboxes):
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        if float('{:.2f}'.format(score[idx])) > .85:
                            # name = names[results[idx]+1]
                            name = names[0]
                            print("if score {:.3f}".format(score[idx]))
                            print("if name %s" %name)
                        else:
                            # name = names[0]
                            name = names[results[idx]+1]
                            print("else score {:.3f}".format(score[idx]))
                            print("else name %s" %name)
                        # frame = draw_box_name(bbox, name, frame)
                        # frame1 = rescale_video_img(frame,0.5)
                frame = draw_box_name(bbox, name, frame)
                
            except:
                pass    
            video_recording.write(frame)
            cv2.imshow('Arc Face Recognizer', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                
    cap.release()
    video_recording.release()
    cv2.destroyAllWindows()



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
    
    

   


