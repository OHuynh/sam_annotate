import tkinter as tk


class Labeler:
    """
    GUI class which is displayed when the start and end frames are annotated
    """
    def __init__(self, annotator_callback):
        self.annotator_callback = annotator_callback

        self._window = tk.Tk()
        self._window.title("Labeler")
        self._window.geometry('325x250')
        self._window.configure(background="gray")

        classes = ('Autonomous Shuttle', 'Shuttle', 'Car', 'Pedestrian')

        classes_var = tk.StringVar(value=classes)
        self._listbox = tk.Listbox(self._window, listvariable=classes_var, height=len(classes))
        self._listbox.pack()

        b = tk.Button(self._window, text="Save", command=lambda: self.save())
        b.pack()

        b = tk.Button(self._window, text="Cancel", command=lambda: self.cancel())
        b.pack()

        self._window.mainloop()

    def save(self):
        label = self._listbox.curselection()
        label = label if label else 0
        self.annotator_callback(True, label, 0)
        self._window.destroy()

    def cancel(self):
        self.annotator_callback(False, 0, 0)
        self._window.destroy()
