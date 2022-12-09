import os
import json
import glob
import argparse

HOME_PATH = os.path.expanduser("~")
DATA_DIR_PATH = os.path.join(HOME_PATH, "input/data")


def main(args):
    images_path = os.path.join(args.dataset_path, 'images')
    output_dir_path = os.path.join(args.dataset_path, 'ufo')
    output_path = os.path.join(output_dir_path, 'train.json')
    file_data = dict()
    file_data["images"] = dict()

    # modify each image's json file 
    for name in sorted(glob.glob(os.path.join(images_path,'*'))):
        temp = dict() 

        tmp = name.split('.')
        tmp[-1] = 'json'

        output_name = name.split('/')[-1]

        json_name = ".".join(tmp).replace('images','legacy_annotations')

        with open(json_name,'r', encoding='UTF8') as f:
            json_data = json.load(f)
            
        temp["img_h"] = json_data["images"][0]["height"]
        temp["img_w"] = json_data["images"][0]["width"]
        temp["words"] = dict()
        temp["tags"] = []
        
        i = 0
        for ann in json_data["annotations"]:
            if(len(ann["bbox"])!=4):
                continue
            
            x,y,w,h = ann["bbox"]
            if ann["bbox"][0] is None:
                continue 
                
            temp["words"][i] = {
                "transcription":ann["text"],
                "language": ["ko"]}
            
            if ann["text"] == "xxx":
                temp["words"][i]["illegibility"]=True
            else:
                temp["words"][i]["illegibility"]=False
        

            if json_data["metadata"][0]["wordorientation"] == "가로":
                temp["words"][i]["orientation"] = "Horizontal"
            elif json_data["metadata"][0]["wordorientation"] == "세로":
                temp["words"][i]["orientation"] = "Vertical"
            else:
                temp["words"][i]["orientation"] = "Irregular"

            point = [[x,y],
                    [x+w,y],
                    [x+w,y+h],
                    [x,y+h]]
            temp["words"][i]["points"] = point
            temp["words"][i]["word_tags"]=None
            i+=1
            
        file_data["images"][output_name] = temp

    try:
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
    except OSError:
        print("Error: Failed to create the directory.")

    # save converted ufo format
    with open(output_path, 'w') as f:  
        json.dump(file_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default=os.path.join(DATA_DIR_PATH, 'AIHub'))
    args = parser.parse_args()

    main(args)