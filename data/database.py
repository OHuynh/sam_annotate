import cv2
import os
import json
from datetime import date, datetime, timedelta

from data.sequence_bound import SequenceBound
from gui.labeler import Labeler

from detector.labelling import track_object_with_YOLO, bounding_box

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

    def interpolate(self, cap):
        for obj in self.database:
            class_obj = self.database[obj][0]
            for seq in self.database[obj][1]:
                for sub_seq_idx, sub_seq in enumerate(seq.sub_sequence):
                    # do not re process if it contains frames
                    if len(sub_seq.time_markers):
                        continue
                    print(f'Interpolation for object : {obj}, sub sequence {sub_seq_idx + 1} / {len(seq.sub_sequence)}')
                    initial_frame = seq.time_markers[sub_seq_idx]
                    final_frame = seq.time_markers[sub_seq_idx + 1]
                    initial_top_left = seq.bb[sub_seq_idx][0]
                    initial_bottom_right = seq.bb[sub_seq_idx][1]
                    final_top_left = seq.bb[sub_seq_idx + 1][0]
                    final_bottom_right = seq.bb[sub_seq_idx + 1][1]

                    initial_bb = bounding_box(initial_frame, initial_top_left[0], initial_top_left[1],
                                              initial_bottom_right[0], initial_bottom_right[1], 1.0, class_obj,
                                              Labeler.classes[0])
                    final_bb = bounding_box(final_frame, final_top_left[0], final_top_left[1],
                                            final_bottom_right[0], final_bottom_right[1], 1.0, class_obj,
                                            Labeler.classes[0])
                    df = track_object_with_YOLO(cap, initial_frame, final_frame, initial_bb, final_bb,
                                                self._detection_model)
                    # fill the subsequence with dataframe
                    for _, row in df.iterrows():
                        sub_seq.insert_frame(time=row['frame'],
                                             top_left=(row['xmin'], row['ymin']),
                                             bottom_right=(row['xmax'], row['ymax']),
                                             type_traj=row['method'],
                                             idx=len(sub_seq.time_markers))

        print(f'Interpolation done.')

    def save_coco_format_json(self, cap):
        """
        # This function should have a fully generated dataset as input (with the interpolation process).
        # A dummy generation is temporary implemented here for testing purpose.
        """
        path_img = os.path.join(self._output_path, 'images')
        if not os.path.exists(path_img):
            os.mkdir(path_img)
        fps = cap.get(cv2.CAP_PROP_FPS)
        filename = os.path.basename(self._video_name)
        begin_date = datetime.strptime(filename[filename.find('_') + 1:], '%Y%m%d_%H%M%S.mp4')
        info = {'version': 'Test',
                'description': 'Dummy data',

                'year': date.today().year,
                'date_created': '', #datetime.today(),
                'contributor' : 'Olivier Huynh'}

        images = []
        categories = []
        for idx, label in enumerate(Labeler.classes):
            category = {'id': idx + 1,
                        'name': label}
            categories.append(category)
        
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
                            marker_annotated = True
                            annotations_found = True
                            annotation = {'id': str(len(annotations)),
                                          'image_id': str(len(images)),
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
                                                  'image_id': str(len(images)),
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                assert ret, 'Invalid annotation'

                saved_path = os.path.join(path_img, str(len(images)) + '.png')
                cv2.imwrite(saved_path, frame)

                date_captured = begin_date + timedelta(seconds=frame_idx / float(fps))
                image = {
                    'id': str(len(images)),
                    'file_name': str(len(images)) + '.png',
                    'location': path_img,
                    # Optional
                    'width': frame.shape[1],
                    'height': frame.shape[0],
                    'date_captured': date_captured.strftime('%Y-%m-%d %H:%M:%S'),
                    'seq_id': '',
                    'seq_num_frames': 0,
                    'frame_num': 0
                }
                images.append(image)

        output = {'info': info,
                  'images': images,
                  'categories': categories,
                  'annotations': annotations}

        output_dump = json.dumps(output)
        with open(os.path.join(self._output_path, 'annotations_coco.json'), 'w') as outfile:
            outfile.write(output_dump)

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
