from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import _init_paths

import os
import cv2

from opts_flir import opts_flir
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
fps = 0


#Implement the function to calculate the IOU
def iou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max(xB - xA, 0) * max(yB - yA, 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def demo(opt):
  global fps
  #opt.debug_dir = 'demo_res/'
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    fps = cam.get(cv2.CAP_PROP_FPS)

    detector.pause = False
    while cam.isOpened():
        _, img = cam.read()
        #cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      bboxes = ret['results']
      #print(bboxes)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      #print(ret['results'][6][0][4])
      #Removing the null/none scored class from the score dictionary(ret)
      dic_out = {x:y for x,y in ret['results'].items() if y.any()}
      #print(dic_out)
      #Removing the predicted scores<0.5 classes - ret is a value sorted dictionary so, y[0] has the largest prediction score object of the class
      new_ret = {x:y for x,y in dic_out.items() if y[0][4]>0.3}
      
      for k,s in new_ret.items():
        indx=[]
        for j in range(len(s)):
          if s[j][-1]<0.3:
            indx.append(j)
        s=np.delete(s,indx,axis=0)

        new_ret[k]=s

      #print(new_ret)
      count=0
      my_dict={}
      for i in new_ret.items():
        #print(i[0])
        for j in i[1]:
          if(j[4]>0.3):
            #print(j[4])
            count+=1
          #else:
           # new_ret = np.delete(new_ret, 1,axis=0)
        #print("count",count)  
        my_dict[i[0]]=count
        count=0 
      print(my_dict)
      #print(new_ret)
      if 13 in my_dict.keys():
        print("Buoy is floating")
      if (6 in my_dict.keys() and (8 in my_dict.keys() or 9 in my_dict.keys())):
        print("Help Requesting")
      if ( (10 in my_dict.keys() and 6 in my_dict.keys()) or 10 in my_dict.keys()):
        print("Jet-Skiing")
      if (5 in my_dict.keys() and (6 in my_dict.keys() or 14 in my_dict.keys()or 7 in my_dict.keys())):
        for k,s in new_ret.items():
          if k==5:
            for j in range(0,my_dict[5]):
              #print(j)
              boxV=[s[j][0],s[j][1],s[j][2],s[j][3]]
              #print (boxV)
              for p,m in new_ret.items():
                #print(k,s)
                if p==7 or p==14 or p==6:
                    for i in range(0,my_dict[p]):
                        boxP=[m[i][0],m[i][1],m[i][2],m[i][3]]
                        #print (boxP)
                        #print(iou(boxV,boxP))
                        if(iou(boxV,boxP)>0 and iou(boxV,boxP)<=1 ):        #####################################################
                            a=iou(boxV,boxP)
                            #print(iou(boxV,boxP))
                            #Scount+=1
                            print("Surfing")
      if 6 in my_dict.keys():
        print("Swimming")
      if (14 in my_dict.keys() and 3 in my_dict.keys()):   #TODO: Implement for other vehicles, this code is for kayak(our dataset only have that #vessel)
        for k,s in new_ret.items():
        #print(k,s)
          if k==3:
            for j in range(0,my_dict[3]):
              Scount=0
              boxV=[s[j][0],s[j][1],s[j][2],s[j][3]]
              #print (boxV)
              for p,m in new_ret.items():
                #print(k,s)
                if p==14:
                    for i in range(0,my_dict[14]):
                        boxP=[m[i][0],m[i][1],m[i][2],m[i][3]]
                        #print (boxP)
                        #print(iou(boxV,boxP))
                        if(iou(boxV,boxP)>0 and iou(boxV,boxP)<1 ):        #####################################################
                            a=iou(boxV,boxP)
                            #print(iou(boxV,boxP))
                            Scount+=1

              if Scount>0:
                print("Human count in Kayak is :" + str(Scount)+ "  >>>>>  If the count is suspicious make an alert on security")
                
                # Make alert notification to the security
              Scount=0
      
if __name__ == '__main__':
  opt = opts_flir().init()


  debugdir = 'demo/'+ opt.demo.split('.')[0].split('/')[-1]
  if not os.path.isdir(debugdir):
    os.makedirs(debugdir)
  opt.debug_dir = debugdir
  print ("Debug Directory: ",opt.debug_dir)

  try:
    demo(opt)
    print(fps)
  except:
    print(fps)