from data.sequence_bound import SequenceBound

from gui.labeler import Labeler


class Database:
    def __init__(self):
        self.database = {}

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
            else:
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
