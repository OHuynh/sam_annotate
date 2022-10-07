def get_tl_br(pt1, pt2):
    top_left = (min(pt1[0], pt2[0]),
                min(pt1[1], pt2[1]))
    bottom_right = (max(pt1[0], pt2[0]),
                    max(pt1[1], pt2[1]))
    return top_left, bottom_right
