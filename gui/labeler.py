import tkinter as tk
from enum import Enum


class LabelerAction(Enum):
    SAVE = 1
    CONTINUE = 2
    CANCEL = 3


class LabelerAction(Enum):
    SAVE = 1
    CONTINUE = 2
    CANCEL = 3

class Labeler:
    classes = ('Autonomous Shuttle', 'Shuttle', 'Car', 'Pedestrian')
    trajectories_type = ('Linear', 'Circular', 'Bezier')
    """
    GUI class which is displayed when the start and end frames are annotated
    """
    def __init__(self, annotator_callback):
        self.annotator_callback = annotator_callback

        self._window = tk.Tk()
        self._window.title("Labeler")
        self._window.geometry('325x250')
        self._window.configure(background="gray")


        classes_var = tk.StringVar(value=self.classes)
        self._classes_list = tk.Listbox(self._window, listvariable=classes_var, height=len(self.classes))
        self._classes_list.pack()

        trajectories_type_var = tk.StringVar(value=self.trajectories_type)
        self._trajectory_list = tk.Listbox(self._window,
                                           listvariable=trajectories_type_var,
                                           height=len(self.trajectories_type))
        self._trajectory_list.pack()

        b = tk.Button(self._window, text="Save", command=lambda: self.save())
        b.pack()

        b = tk.Button(self._window, text="Continue", command=lambda: self.keep_on())
        b.pack()

        b = tk.Button(self._window, text="Cancel", command=lambda: self.cancel())
        b.pack()

        self._window.mainloop()

    def save(self):
        label = self._classes_list.curselection()
        label = label[0] if label else 0
        trajectory = self._trajectory_list.curselection()
        trajectory = trajectory[0] if trajectory else 0
        self.annotator_callback(LabelerAction.SAVE, label, 0, trajectory)
        self._window.destroy()

    def keep_on(self):
        label = self._classes_list.curselection()
        label = label[0] if label else 0
        trajectory = self._trajectory_list.curselection()
        trajectory = trajectory[0] if trajectory else 0
        self.annotator_callback(LabelerAction.CONTINUE, label, -1, trajectory)
        self._window.destroy()

    def cancel(self):
        self.annotator_callback(LabelerAction.CANCEL, 0, 0, 0)
        self._window.destroy()
