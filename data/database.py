import cv2
import os
import json
from datetime import date, datetime, timedelta
import pandas as pd

from data.sequence_bound import SequenceBound
from gui.labeler import Labeler

#from detector.yolo.yolov7 import track_object_with_YOLO, bounding_box, list_of_classes, label_coco_map

class Database:
    def __init__(self, video_name, output_path, detection_model):
        self.database = {}
        self._video_name = video_name
        self._output_path = output_path
        self._detection_model = detection_model

    def add_sample(self, sequence_bb, label, obj_id):
        sequence = SequenceBound(sequence_bb, add_sub_seq=True)

        # assign a new object
        if obj_id == -1:
            obj_id = len(self.database)

        valid_sequence = True
        error_message = ''
        error_one_sample_alone = 'Annotation shall be a sequence of at least 2 boxes or be between two annotation.'
        if obj_id in self.database:
            sequence_to_merge = []
            indices_to_pop = []
            for idx, other_seq in enumerate(self.database[obj_id][1]):
                intersect = not ((sequence.time_markers[0] <= sequence.time_markers[-1] <
                                  other_seq.time_markers[0] <= other_seq.time_markers[-1])
                                 or
                                 (other_seq.time_markers[0] <= other_seq.time_markers[-1] <
                                  sequence.time_markers[0] <= sequence.time_markers[-1]))
                if intersect:
                    if not len(sequence_to_merge):
                        sequence_to_merge.append(sequence)
                    sequence_to_merge.append(other_seq)
                    indices_to_pop.append(idx)

            if len(sequence_bb) == 1 and not len(sequence_to_merge):
                valid_sequence = False
                error_message = error_one_sample_alone
            elif len(sequence_to_merge):
                for idx in indices_to_pop[::-1]:
                    self.database[obj_id][1].pop(idx)
                new_seq = []
                for seq in sequence_to_merge:
                    new_seq += seq.sequence
                sequence = SequenceBound(new_seq, add_sub_seq=True)
        else:
            if len(sequence_bb) == 1:
                valid_sequence = False
                error_message = error_one_sample_alone
            else:
                self.database[obj_id] = [label, []]

        if valid_sequence:
            self.database[obj_id][1].append(sequence)
        else:
            print(error_message)
        print(self)

    def load_json(self, path):
        if not path:
            filename = os.path.splitext(os.path.basename(self._video_name))[0]
            path = os.path.join(self._output_path, filename + '.json')
        if not os.path.exists(path):
            return
        with open(path, 'r') as openfile:
            data = json.load(openfile)
        # map data to sequence objects
        for obj_id_str in data:
            obj_id = int(obj_id_str)
            for sequence in data[obj_id_str][1]:
                self.add_sample(sequence['sequence'], int(data[obj_id_str][0]), obj_id)
                if sequence['sub_sequence']:
                    sub_sequence = []
                    for sub_seq in sequence['sub_sequence']:
                        sub_sequence.append(SequenceBound(sub_seq))
                    self.database[obj_id][1][-1].sub_sequence = sub_sequence

    def get_list_str_obj(self):
        return [f'ID {obj_id} Class {Labeler.classes[self.database[obj_id][0]]}' for obj_id in self.database]

    def __str__(self):
        string = ''
        for annot in self.database:

            string += f'id : {annot} class : {Labeler.classes[self.database[annot][0]]} \n'
            for sequence in self.database[annot][1]:
                string += str(sequence)
                string += "\n"

            string += '\n ------------------------------------------- \n'
        return string

    def interpolate(self):
        for obj in self.database:
            label = self.database[obj][0]
            for seq in self.database[obj][1]:
                self._detection_model.interpolate(seq, label)
        print(f'Interpolation done.')

    def to_coco_format_json(self, cap, counter_img):
        """
        This function returns the annotations with coco-format
        """
        images = []
        annotations = []

        # creation of dummy data using only the bounding boxes annotated of the sequence
        nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for frame_idx in range(int(nb_frames)):
            annotations_found = False
            for obj_id in self.database:
                for sequence in self.database[obj_id][1]:
                    marker_annotated = False
                    for time_marker, bb in zip(sequence.time_markers, sequence.bb):
                        if time_marker == frame_idx:
                            annotations_found = True
                            marker_annotated = True
                            annotation = {'id': str(len(annotations)),
                                          'image_id': counter_img,
                                          'category_id': self.database[obj_id][0] + 1,  # coco format
                                          # reserves 0 for 'empty'
                                          'bbox': [bb[0][0],  # x
                                                   bb[0][1],  # y
                                                   bb[1][0] - bb[0][0],  # width
                                                   bb[1][1] - bb[0][1]],  # height
                                          'sequence_level_annotation': False
                                          }
                            annotations.append(annotation)
                    # priority on time marker if there it is overlapped in time
                    if not marker_annotated:
                        for sub_seq in sequence.sub_sequence:
                            for time_marker, bb in zip(sub_seq.time_markers, sub_seq.bb):
                                if time_marker == frame_idx:
                                    annotations_found = True
                                    annotation = {'id': str(len(annotations)),
                                                  'image_id': counter_img,
                                                  'category_id': self.database[obj_id][0] + 1,  # coco format
                                                  # reserves 0 for 'empty'
                                                  'bbox': [bb[0][0],  # x
                                                           bb[0][1],  # y
                                                           bb[1][0] - bb[0][0],  # width
                                                           bb[1][1] - bb[0][1]],  # height
                                                  'sequence_level_annotation': False
                                                  }
                                    annotations.append(annotation)
            if annotations_found:
                images.append((frame_idx, counter_img))
                counter_img += 1

        return annotations, images

    def save_json(self):
        filename = os.path.splitext(os.path.basename(self._video_name))[0]
        output = {}
        for obj_id in self.database:
            output[obj_id] = [self.database[obj_id][0], []]
            for sequence in self.database[obj_id][1]:
                sequence_bound = {'sequence': sequence.sequence, 'sub_sequence': []}
                for sub_seq in sequence.sub_sequence:
                    sequence_bound['sub_sequence'].append(sub_seq.sequence)
                output[obj_id][1].append(sequence_bound)

        output_dump = json.dumps(output)
        with open(os.path.join(self._output_path, filename + '.json'), 'w') as outfile:
            outfile.write(output_dump)
        print('Annotations saved.')

    def save_yolo_format_txt(self, cap, save_img=False):
        """
        There are two outputs from this function:
            1) Annotation data in YOLO format (txt files) for fine tunning / transfer learning process
                path where labels (txt files) are saved: self._output_path/labelling/file_name/labels
                txt file format: <object-class> <x> <y> <width> <height>
                optional, path where images are saved: self._output_path/labelling/file_name/images
            2) YOLO detection data (txt files) generated during interpolation process
                path where txt files are saved: self._output_path/runs/detect/file_name/labels
                txt file format: <object-class> <x> <y> <width> <height> <confidence>
        """
        # name for the txt files: videoFileName_frame_number.txt
        videoFileName = os.path.splitext(os.path.basename(self._video_name))[0]
        # Path used to save txt files
        # labels will be saved inside the folder 'labelling/file_name/labels'
        pathToLabels = os.path.join(self._output_path, 'labelling')
        if not os.path.exists(pathToLabels):
            os.mkdir(pathToLabels)
        pathToLabels = os.path.join(pathToLabels, videoFileName)
        if not os.path.exists(pathToLabels):
            os.mkdir(pathToLabels)
        pathToLabels = os.path.join(pathToLabels, 'labels')
        if not os.path.exists(pathToLabels):
            os.mkdir(pathToLabels)
        # Path used to save images
        # images will be saved inside the folder 'labelling/file_name/images'
        pathToImages = os.path.join(self._output_path, 'labelling')
        if not os.path.exists(pathToImages):
            os.mkdir(pathToImages)
        pathToImages = os.path.join(pathToImages, videoFileName)
        if not os.path.exists(pathToImages):
            os.mkdir(pathToImages)
        pathToImages = os.path.join(pathToImages, 'images')
        if not os.path.exists(pathToImages):
            os.mkdir(pathToImages)
        # extension of the files (txt)
        labels_extension = '.txt'
        # dataframe used to write txt files
        df_columns = ["frame", "class", "x_center", "y_center", "width", "height"]
        df = pd.DataFrame(columns=df_columns)
        # information about the video and frame
        nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # for each frame inside the sequences, create a row inside the dataframe df
        for obj_id in self.database:
            for sequence in self.database[obj_id][1]:
                for sub_seq in sequence.sub_sequence:
                    for i in range(len(sub_seq.sequence)):
                        xmin, xmax = sub_seq.sequence[i][1][0], sub_seq.sequence[i][2][0]
                        ymin, ymax = sub_seq.sequence[i][1][1], sub_seq.sequence[i][2][1]
                        df.loc[len(df.index)] = [int(sub_seq.sequence[i][0]), # frame number
                                                 int(self.database[obj_id][0]),  # class
                                                 (((xmax - xmin) / 2) + xmin) / frame_width, # x_center (normalized)
                                                 (((ymax - ymin) / 2) + ymin) / frame_height,# y_center (normalized)
                                                 (xmax - xmin) / frame_width,  # width
                                                 (ymax - ymin) / frame_height]  # height
                        if save_img:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sub_seq.sequence[i][0]))
                            ret, frame = cap.read()
                            assert ret, 'Invalid annotation'
                            img_file = pathToImages + '/' + videoFileName + '_' + str(int(sub_seq.sequence[i][0])) + '.png'
                            cv2.imwrite(img_file, frame)
        # once all sequences are stored inside dataframe df, df is saved in txt format
        for frame_idx in range(int(nb_frames)):
            # find all registers inside 'df' corresponding to the actual frame
            aux = df[df['frame'] == frame_idx]
            # if there is at least one annotated object
            if len(aux.index):
                # file to write
                file_txt = pathToLabels + '/' + videoFileName + "_" + str(frame_idx) + labels_extension
                # auxiliar variable used to know if it is the first line to write
                first_time = True
                # write new_box inside corresponding txt file
                for element in list(aux.index):
                    # if it is the first row to write then overwrite the file
                    if first_time:
                        opt = 'w'
                        first_time = False
                    # otherwise, add the line at the end of the file
                    else:
                        opt = 'a'
                    with open(file_txt, opt) as fp:
                        new_line = str(int(aux.loc[element,'class'])) + ' ' + \
                                   str(float(aux.loc[element,'x_center'])) + ' ' + \
                                   str(float(aux.loc[element,'y_center'])) + ' ' + \
                                   str(float(aux.loc[element,'width'])) + ' ' + \
                                   str(float(aux.loc[element,'height'])) + '\n'
                        fp.write(new_line)
                        fp.close()
        print(f'labels saved in yolo format. Path: {pathToLabels}')
        ###############################
        # Save data from YOLO Detection
        # detection data from yolo is saved in self._data_from_yolo
        # then this data is taken from self._data_from_yolo and saved in txt files (as yolo's output)
        # pathToYOLOData = Path used to save detection carried out with YOLO (txt files)
        # labels will be saved inside the folder 'runs/detect/file_name/labels'
        pathToYOLOData = os.path.join(self._output_path, 'runs')
        if not os.path.exists(pathToYOLOData):
            os.mkdir(pathToYOLOData)
        pathToYOLOData = os.path.join(pathToYOLOData, 'detect')
        if not os.path.exists(pathToYOLOData):
            os.mkdir(pathToYOLOData)
        pathToYOLOData = os.path.join(pathToYOLOData, videoFileName)
        if not os.path.exists(pathToYOLOData):
            os.mkdir(pathToYOLOData)
        pathToYOLOData = os.path.join(pathToYOLOData, 'labels')
        if not os.path.exists(pathToYOLOData):
            os.mkdir(pathToYOLOData)
        # for each frame number (not very efficient)
        for frame_idx in range(int(nb_frames)):
            # only write the yolo file if the .txt file doesn't exist (meaning that yolo has not been executed before)
            file_txt = pathToYOLOData + '/' + videoFileName + "_" + str(frame_idx) + labels_extension
            if not (os.path.isfile(file_txt)):
                # find all registers inside '_data_from_yolo' corresponding to the actual frame
                aux = self._data_from_yolo[self._data_from_yolo['frame'] == frame_idx]
                # if there are detected objects in _data_from_yolo
                if len(aux.index):
                    # write new_box inside corresponding txt file
                    with open(file_txt, "a") as fp:
                        for element in list(aux.index):
                            width = (aux.loc[element, 'xmax'] - aux.loc[element, 'xmin']) / frame_width # normalized
                            height = (aux.loc[element, 'ymax'] - aux.loc[element, 'ymin']) / frame_height  # normalized
                            x_center = (aux.loc[element, 'xmin'] / frame_width) + (width/2) # normalized
                            y_center = (aux.loc[element, 'ymin'] / frame_height) + (height / 2)  # normalized
                            new_line = str(int(aux.loc[element, 'class'])) + ' ' + \
                                       str(x_center) + ' ' + \
                                       str(y_center) + ' ' + \
                                       str(width) + ' ' + \
                                       str(height) + ' ' + \
                                       str(aux.loc[element, 'confidence']) + '\n'
                            fp.write(new_line)
                        fp.close()
        print(f'Data from yolo detection saved in path: {pathToYOLOData}')

