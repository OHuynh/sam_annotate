from data.sequence_bound import SequenceBound

from gui.labeler import Labeler


class Database:
    def __init__(self):
        self._database = {}

    def add_sample(self, sequence_bb, label, obj_id):
        sequence = SequenceBound(sequence_bb)

        # assign a new object
        if obj_id == -1:
            obj_id = len(self._database)

        valid_sequence = True
        if obj_id in self._database:
            for other_seq in self._database[obj_id][1]:
                not_intersect = (sequence.time_markers[0] < sequence.time_markers[1] <
                                 other_seq.time_markers[0] < other_seq.time_markers[1]) \
                                or \
                                (other_seq.time_markers[0] < other_seq.time_markers[1] <
                                 sequence.time_markers[0] < sequence.time_markers[1])
                if not not_intersect:
                    valid_sequence = False
                    break
        else:
            self._database[obj_id] = [label, []]

        if valid_sequence:
            self._database[obj_id][1].append(sequence)
        else:
            print('Discard this annotation because it intersects another previously annotated sequence.')
        print(self)

    def get_list_str_obj(self):
        return [f'ID {obj_id} Class {Labeler.classes[self._database[obj_id][0]]}' for obj_id in self._database]

    def __str__(self):
        string = ''
        for annot in self._database:

            string += f'id : {annot} class : {Labeler.classes[self._database[annot][0]]} \n'
            for sequence in self._database[annot][1]:
                string += str(sequence)
                string += "\n"

            string += '\n ------------------------------------------- \n'
        return string
