import json
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import threading
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, filepath, update_callback):
        super().__init__()
        self.filepath = filepath
        self.update_callback = update_callback

    def on_modified(self, event):
        if os.path.abspath(event.src_path) == os.path.abspath(self.filepath):
            self.update_callback()

class LivePlotter:
    def __init__(self, master):
        self.master = master
        self.filepath = None
        self.observer = None

        self.btn_load = btn = tk.Button(root, text="Load simulation_log.json and plot", command=self.load_file, width=30, height=3)
        self.btn_load.pack(expand=True, fill='both')

        # Setup matplotlib for interactive plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.tight_layout()

    def load_file(self):
        self.filepath = filedialog.askopenfilename(title="Open simulation_log.json",
                                                   filetypes=[("JSON files", "*.json")])
        if self.filepath:
            self.start_watching()
            self.update_plot()  # Initial plot

    def start_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
        event_handler = FileChangeHandler(self.filepath, self.update_plot)
        self.observer = Observer()
        directory = os.path.dirname(self.filepath)
        self.observer.schedule(event_handler, path=directory, recursive=False)
        self.observer.start()

    def update_plot(self):
        time.sleep(1)
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)

            gens = [entry['generation'] for entry in data]
            populations = [entry['start_population'] for entry in data]
            food_eaten = [entry['food_eaten'] for entry in data]
            survived = [entry['survived'] for entry in data]
            success_percent = [entry['success_percent'] for entry in data]

            # Clear previous plots
            for ax in self.axes.flatten():
                ax.clear()

            # Plot data
            self.axes[0, 0].plot(gens, populations, label="Population")
            self.axes[0, 0].set_xlabel("Generation")
            self.axes[0, 0].set_ylabel("Population")
            self.axes[0, 0].grid(True)
            self.axes[0, 0].legend()

            self.axes[0, 1].plot(gens, food_eaten, label="Food Eaten", color="orange")
            self.axes[0, 1].set_xlabel("Generation")
            self.axes[0, 1].set_ylabel("Food Eaten")
            self.axes[0, 1].grid(True)
            self.axes[0, 1].legend()

            self.axes[1, 0].plot(gens, survived, label="Survivors", color="green")
            self.axes[1, 0].set_xlabel("Generation")
            self.axes[1, 0].set_ylabel("Survivors")
            self.axes[1, 0].grid(True)
            self.axes[1, 0].legend()

            self.axes[1, 1].plot(gens, success_percent, label="Success %", color="red")
            self.axes[1, 1].set_xlabel("Generation")
            self.axes[1, 1].set_ylabel("Success %")
            self.axes[1, 1].grid(True)
            self.axes[1, 1].legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"Error updating plot: {e}")

    def cleanup(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Live Simulation Log Viewer")
    app = LivePlotter(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()