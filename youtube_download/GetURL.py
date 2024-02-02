from common import get_class_list, date_range_finish, date_range_start
from youtube_search import YoutubeSearch
import os
import json
import urllib.request


def main():
    class_list = get_class_list()

    search_num_per_class = [20 for i in range(len(class_list))]
    print(search_num_per_class)

    total_video = [0 for i in range(len(class_list))]

    for sdate, edate in zip(date_range_start, date_range_finish):

        for iidx, class_name in enumerate(class_list):

            if total_video[iidx] > 300:
                print("Skip the %s. its enough."%(class_name))
                continue

            if not os.path.isdir(os.path.join("./videos_thumbnail", class_name)):
                os.makedirs(os.path.join("./videos_thumbnail", class_name))

            q = class_name+'&sp=EgYQARgBMAE%253D' ## change this search query to proper one.

            print(q)

            n = search_num_per_class[iidx]
            results = YoutubeSearch(q, max_results=n).to_dict()

            if results == None:
                print("Not enough search results. Skip the current date range..")
                continue

            for idx, single_result in enumerate(results):
                id = single_result['id']
                inner_path = os.path.join("./videos_thumbnail", class_name, id)

                if not os.path.isdir(inner_path):
                    os.makedirs(inner_path)
                    total_video[iidx] += 1
                else:
                    continue
                
                thumbnail_url = single_result['thumbnails'][-1]
                urllib.request.urlretrieve(thumbnail_url, os.path.join(inner_path, "thumbnail_%s.png"%(id)))

                with open(os.path.join(inner_path, "info_dict.json"), 'w') as fp:
                    json.dump(single_result, fp)

            print("Processed videos : ",total_video, sum(total_video))



if __name__ == "__main__":
    main()
    