import os
import glob
import json
from yt_dlp import YoutubeDL
from tqdm import tqdm

thumbnail_path_list={}

root = "/data01/kinux98/SSSCWEB/youtube_download" # change to proper one.

image_dir = os.path.join(root, "videos_thumbnail")
class_list_dir = os.listdir(image_dir)
cnt = 0
print(class_list_dir)
for scls in tqdm(class_list_dir, total=len(class_list_dir), leave=False):
    pathj = os.path.join(image_dir, scls)
    print(pathj)
    sub_class_list_dir = os.listdir(pathj)
    print(sub_class_list_dir)

    for sscls in tqdm(sub_class_list_dir, total=len(sub_class_list_dir), leave=False):
        spathj = os.path.join(pathj, sscls, "*.json")
        print(spathj)
        jfilep = glob.glob(spathj)[0]
        print(jfilep)

        SAVE_PATH = os.path.join(pathj, sscls)
        if os.path.isfile(os.path.join(SAVE_PATH, sscls+".mp4")):
            cnt += 1
            continue
        ydl_opt = {
            "format" : "bestvideo[ext=mp4]/mp4",
            "keepvideo" : 'True',
            "noplaylist":'True',
            "outtmpl" : SAVE_PATH + '/%(id)s.%(ext)s'
        }
        success=False
        while not success:
            try:
                with open(jfilep) as jfile:
                    data = json.load(jfile)
                    url_suffix = data['url_suffix']
                    print(data)
                    print("https://youtube.com"+data['url_suffix'])
                    with YoutubeDL(ydl_opt) as ydl:
                        ydl.download(["https://youtube.com"+data['url_suffix']])
                success=True
            except:
                success=False
        cnt +=1 
        print("Total : ", cnt)
            




