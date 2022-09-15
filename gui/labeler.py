import tkinter as tk
from enum import Enum

from data.sequence_bound import TrajectoryTypes


class LabelerAction(Enum):
    SAVE = 1
    CONTINUE = 2
    DISCARD = 3
    CANCEL = 4


class Labeler:
    #classes = ('Pedestrian', 'Car', 'Autonomous Shuttle', 'Shuttle')
    classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck')
    """
    GUI class which is displayed when the start and end frames are annotated
    """
    def __init__(self, annotator_callback, database):
        self.annotator_callback = annotator_callback

        self._window = tk.Tk()
        self._window.title("Labeler")
        self._window.geometry('400x400')
        self._window.configure(background="gray")
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=3)

        classes_var = tk.StringVar(value=self.classes)
        self._classes_list = tk.Listbox(self._window, listvariable=classes_var, height=len(self.classes),
                                        exportselection=0)
        self._classes_list.grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        trajectory_types_arr = [type_traj.name for type_traj in TrajectoryTypes]
        trajectories_type_var = tk.StringVar(value=trajectory_types_arr)
        self._trajectory_list = tk.Listbox(self._window,
                                           listvariable=trajectories_type_var,
                                           height=len(trajectory_types_arr),
                                           exportselection=0)
        self._trajectory_list.grid(column=0, row=1, columnspan=2, sticky=tk.W, padx=5, pady=5)

        list_ids = database.get_list_str_obj()
        list_ids.insert(0, 'New ID')
        id_obj_var = tk.StringVar(value=list_ids)
        self._id_obj_list = tk.Listbox(self._window,
                                       listvariable=id_obj_var,
                                       height=len(list_ids),
                                       width=30,
                                       exportselection=0)
        self._id_obj_list.grid(column=2, row=0, rowspan=2, sticky=tk.E, padx=5, pady=5)

        b = tk.Button(self._window, text="Save", command=lambda: self.save())
        b.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)

        b = tk.Button(self._window, text="Continue", command=lambda: self.keep_on())
        b.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)

        b = tk.Button(self._window, text="Discard", command=lambda: self.discard())
        b.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

        b = tk.Button(self._window, text="Cancel", command=lambda: self.cancel())
        b.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)

        self._window.mainloop()

    def save(self):
        label = self._classes_list.curselection()
        label = label[0] if label else 0
        trajectory = self._trajectory_list.curselection()
        trajectory = trajectory[0] if trajectory else 0
        id = self._id_obj_list.curselection()
        id = id[0] - 1 if id else -1
        self.annotator_callback(LabelerAction.SAVE, label, id, trajectory)
        self._window.destroy()

    def keep_on(self):
        label = self._classes_list.curselection()
        label = label[0] if label else 0
        trajectory = self._trajectory_list.curselection()
        trajectory = trajectory[0] if trajectory else 0
        id = self._id_obj_list.curselection()
        id = id[0] - 1 if id else -1
        self.annotator_callback(LabelerAction.CONTINUE, label, id, trajectory)
        self._window.destroy()

    def discard(self):
        self.annotator_callback(LabelerAction.DISCARD, 0, -1, 0)
        self._window.destroy()

    def cancel(self):
        self.annotator_callback(LabelerAction.CANCEL, 0, -1, 0)
        self._window.destroy()

