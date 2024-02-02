from audioop import mul
from concurrent.futures import process
import os
import glob
import json
from tqdm import tqdm
from scenedetect import detect, ContentDetector, AdaptiveDetector
import shutil, cv2
from multiprocessing import Pool
import multiprocessing
from functools import partial

root = "/your/path/SSSCWEB/youtube_download"

video_dir = os.path.join(root, "videos_thumbnail_youtube") # load PATH

image_dir = os.path.join(root, "videos_thumbnail_test") # save PATH

class_list_dir = os.listdir(video_dir)
cnt = 0

# save_dict = {}
frame = 0


def do_save(class_list_dir_):
    save_dict_ = {}
    print(class_list_dir_)
    for scls in tqdm(class_list_dir_, total=len(class_list_dir_), leave=False, desc='total'):
        path_save = os.path.join(image_dir, scls)
    
        path_video = os.path.join(video_dir, scls)
        
        sub_class_list_dir = os.listdir(path_video)
    
        save_dict_[scls] = {}
    
        for sscls in tqdm(sub_class_list_dir, total=len(sub_class_list_dir), leave=False, desc='class'):
            spathj = os.path.join(path_video, sscls, "*.mp4")
            print(spathj)
            jfilep = glob.glob(spathj)[0]
    
            print(jfilep.split('/')[-1])
            print("processing : ", jfilep)
    
            save_dict_[scls][sscls] = {}

            if not os.path.exists(os.path.join(path_save, sscls, "frames")):
                os.makedirs(os.path.join(path_save, sscls, "frames"))
    
            scene_list  = detect(video_path= jfilep, detector=ContentDetector(), save_path=os.path.join(path_save, sscls, "frames"))
    
            skip_cnt = 1 # max(min(2, int(fr/8)), 1)
    
            save_dict_[scls][sscls]['skip_cnt'] = skip_cnt
    
            for i, scene in enumerate(scene_list):
                    print('Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                           i+1,
                           scene[0].get_timecode(), scene[0].get_frames(),
                           scene[1].get_timecode(), scene[1].get_frames(),)
                    )
                    save_dict_[scls][sscls][i] = (scene[0].get_frames(), scene[1].get_frames())
    
            print(save_dict_[scls][sscls])

    return save_dict_

process_num = 20 # change to proper one

p = Pool(process_num)

def split(list_a, chunk_size):

  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

chunk_size = len(class_list_dir) // process_num
my_list = class_list_dir
my_list_divided = list(split(my_list, chunk_size))

guess = p.map(do_save, my_list_divided)
p.close()
p.join()
print(guess)

final_list = {}
for single_list in guess:
    final_list |= single_list

# with open("video_scene_parsing_new_mt_test.json", 'w') as fp:
#     json.dump(final_list, fp)