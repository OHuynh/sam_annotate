from utils.geometry import get_tl_br

class SequenceBound:
    def __init__(self, sequence_bb):
        sorted_sequence = sequence_bb

        # because gui appears only until the second
        # store the type of trajectory of the second annot into the first one
        if len(sequence_bb) > 1:
            first_annot = sorted_sequence.pop(0)
            sorted_sequence.append((*first_annot[:-1], sorted_sequence[0][3]))
        sorted_sequence.sort(key=lambda annot: annot[0])
        self.sequence = sorted_sequence
        self.time_markers = []
        self.bb = []
        self.type_traj = []
        for annot in sorted_sequence:
            self.time_markers.append(annot[0])
            self.bb.append((annot[1], annot[2]))
            self.type_traj.append(annot[3])

    def __str__(self):
        return f"{[(time, bb, type_traj) for time, bb, type_traj in zip(self.time_markers, self.bb, self.type_traj)]}"

    def edit(self, point, chunk_idx, new_pos, frame_idx):
        top_left = list(self.bb[chunk_idx][0])
        bottom_right = list(self.bb[chunk_idx][1])
        if point == 0:  # tl
            top_left = new_pos
        elif point == 1:  # tr
            bottom_right[0] = new_pos[0]
            top_left[1] = new_pos[1]
        elif point == 2:  # bl
            top_left[0] = new_pos[0]
            bottom_right[1] = new_pos[1]
        elif point == 3:  # br
            bottom_right = new_pos
        top_left, bottom_right = get_tl_br(top_left, bottom_right)

        self.bb[chunk_idx] = (top_left, bottom_right)
        self.time_markers[chunk_idx] = frame_idx
