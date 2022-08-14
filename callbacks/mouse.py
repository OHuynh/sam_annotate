import cv2
import utils.globals as globals

def draw_bb(action, x, y, flags, *userdata):
    if action == cv2.EVENT_LBUTTONDOWN:
        globals.top_left = (x, y)
        globals.bottom_right = (x, y)
        globals.drawing_rect = True
    elif action == cv2.EVENT_MOUSEMOVE and globals.drawing_rect:
        globals.bottom_right = (x, y)
    elif action == cv2.EVENT_LBUTTONUP:
        globals.bottom_right = (x, y)
        globals.drawing_rect = False
        globals.save_annot = True