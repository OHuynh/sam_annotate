r"""A script to group all the annotations generated from the tool.

Example Usage:
--------------
python generate_coco_dataset.py \
  --input-json-folder path to the folder containing the json files \
  --input-video-folder path to the folder containing the video files \
  --output-path path where the json

"""

import argparse
import cv2
import os
import glob
import json
from datetime import date, datetime, timedelta

from gui.labeler import Labeler
from data import database

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-video-folder", type=str, help="The path to the video to annotate."
)

parser.add_argument(
    "--input-json-folder", type=str, help="The path to the previously generated json annotation to import."
)

parser.add_argument(
    "--output-path", type=str, help="This path will be used to store the coco-formatted json file "
                                    "and the associate images.",
    default='./'
)


if __name__ == "__main__":
    args = parser.parse_args()
    annotation_files = glob.glob(args.input_json_folder + '/Cam*.json')

    path_img = os.path.join(args.output_path, 'images')
    if not os.path.exists(path_img):
        os.mkdir(path_img)
    counter_img = 0
    all_images = []
    all_annotations = []
    for idx, annotation_file in enumerate(annotation_files):
        filename = os.path.splitext(os.path.basename(annotation_file))[0]
        path_to_video = os.path.join(args.input_video_folder, filename + '.mp4')
        if not os.path.exists(path_to_video):
            print(f'{path_to_video} not found.')
            continue
        database = database.Database(path_to_video, args.output_path, None)
        database.load_json(annotation_file)
        cap = cv2.VideoCapture(path_to_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        begin_date = datetime.strptime(filename[filename.find('_') + 1:], '%Y%m%d_%H%M%S')

        annotations, images_to_save = database.to_coco_format_json(cap, counter_img)
        all_annotations = all_annotations + annotations
        for image_to_save in images_to_save:
            frame_idx, counter_img = image_to_save[0], image_to_save[1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            assert ret, 'Invalid annotation'

            saved_path = os.path.join(path_img, str(counter_img) + '.png')
            cv2.imwrite(saved_path, frame)

            date_captured = begin_date + timedelta(seconds=frame_idx / float(fps))
            image = {
                'id': counter_img,
                'file_name': str(counter_img) + '.png',
                'location': path_img,
                # Optional
                'width': frame.shape[1],
                'height': frame.shape[0],
                'date_captured': date_captured.strftime('%Y-%m-%d %H:%M:%S'),
                'seq_id': str(idx),
                'seq_num_frames': 0,
                'frame_num': 0
            }
            all_images.append(image)
        print(f'{idx + 1}/ {len(annotation_files)} videos processed.')
        cap.release()
        cv2.destroyAllWindows()

    info = {'version': 'Database SAM',
            'description': '',
            'year': date.today().year,
            'date_created': str(datetime.today()),
            'contributor': 'SAM Team'}
    categories = []
    for idx, label in enumerate(Labeler.classes):
        category = {'id': idx + 1,
                    'name': label}
        categories.append(category)

    output = {'info': info,
              'images': all_images,
              'categories': categories,
              'annotations': all_annotations}

    output_dump = json.dumps(output)
    with open(os.path.join(args.output_path, 'annotations_coco.json'), 'w') as outfile:
        outfile.write(output_dump)
