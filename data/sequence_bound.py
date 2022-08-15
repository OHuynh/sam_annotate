class SequenceBound:
    def __init__(self, sequence_bb, label, obj_id):
        sorted_sequence = sequence_bb

        # because gui appears only until the second
        # store the type of trajectory of the second annot into the first one
        first_annot = sorted_sequence.pop(0)
        sorted_sequence.append((*first_annot[:-1], sorted_sequence[0][3]))
        sorted_sequence.sort(key=lambda annot: annot[0])

        self.time_markers = []
        self.bb = []
        self.type_traj = []
        for annot in sorted_sequence:
            self.time_markers.append(annot[0])
            self.bb.append((annot[1], annot[2]))
            self.type_traj.append(annot[3])
