from data.sequence_bound import SequenceBound


class Database:
    def __init__(self):
        self._database = {}

    def add_sample(self, sequence_bb, label, obj_id):
        sequence = SequenceBound(sequence_bb, label, obj_id)

        valid_sequence = True
        if obj_id in self._database:
            for other_seq in self._database[obj_id]:
                not_intersect = (sequence.time_markers[0] < sequence.time_markers[1] <
                                 other_seq.time_markers[0] < other_seq.time_markers[1]) \
                                or \
                                (other_seq.time_markers[0] < other_seq.time_markers[1] <
                                 sequence.time_markers[0] < sequence.time_markers[1])
                if not not_intersect:
                    valid_sequence = False
                    break
        if valid_sequence:
            self._database[obj_id].append(sequence)
