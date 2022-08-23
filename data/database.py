import cv2
import os
import json
from datetime import date, datetime

from data.sequence_bound import SequenceBound
from gui.labeler import Labeler


class Database:
    def __init__(self, output_path):
        self.database = {}
        self._output_path = output_path

    def add_sample(self, sequence_bb, label, obj_id):
        sequence = SequenceBound(sequence_bb)

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
                sequence = SequenceBound(new_seq)
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

    def save_coco_format_json(self, cap):
        """
        # This function should have a fully generated dataset as input (with the interpolation process).
        # A dummy generation is temporary implemented here for testing purpose.
        """
        path_img = os.path.join(self._output_path, 'images')
        if not os.path.exists(path_img):
            os.mkdir(path_img)

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
                    for time_marker, bb in zip(sequence.time_markers, sequence.bb):
                        if time_marker == frame_idx:
                            annotations_found = True
                            annotation = {'id': str(len(annotations)),
                                          'image_id': str(len(images)),
                                          'category_id': self.database[obj_id][0] + 1,  # coco format
                                                                                        # reserves 0 for 'empty'
                                          'bbox': [bb[0][0],  # x
                                                   bb[0][1],  # y
                                                   bb[1][0] - bb[0][0],   # width
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

                image = {
                    'id': str(len(images)),
                    'file_name': str(len(images)) + '.png',
                    'location': path_img,
                    # Optional
                    'width': frame.shape[1],
                    'height': frame.shape[0],
                    'date_captured': '', # datetime.today(), # todo compute the date of this frame using the fps
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
