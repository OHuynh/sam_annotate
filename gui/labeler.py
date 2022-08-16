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
    def __init__(self, annotator_callback, database):
        self.annotator_callback = annotator_callback

        self._window = tk.Tk()
        self._window.title("Labeler")
        self._window.geometry('400x250')
        self._window.configure(background="gray")
        self._window.columnconfigure(0, weight=1)
        self._window.columnconfigure(1, weight=3)

        classes_var = tk.StringVar(value=self.classes)
        self._classes_list = tk.Listbox(self._window, listvariable=classes_var, height=len(self.classes),
                                        exportselection=0)
        self._classes_list.grid(column=0, row=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        trajectories_type_var = tk.StringVar(value=self.trajectories_type)
        self._trajectory_list = tk.Listbox(self._window,
                                           listvariable=trajectories_type_var,
                                           height=len(self.trajectories_type),
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

        b = tk.Button(self._window, text="Cancel", command=lambda: self.cancel())
        b.grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)

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

    def cancel(self):
        self.annotator_callback(LabelerAction.CANCEL, 0, -1, 0)
        self._window.destroy()