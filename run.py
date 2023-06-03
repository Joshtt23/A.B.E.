import os
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import torch
from PIL import ImageTk, Image
from main import run_with_gui, LoadingBar
from config import Config
from inspect import getattr_static
import ast
from pathlib import Path
import types


class GpuSelector:
    def __init__(self, root):
        self.root = root
        self.gpu_var = tk.StringVar(self.root)
        self.gpu_menu = None
        self.popup = None

    def select_gpu(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                self.create_gpu_selection_box(gpu_count)
            else:
                self.show_popup("No GPUs available. CUDA is not available.")
        else:
            self.show_popup("CUDA is not available.")

    def create_gpu_selection_box(self, gpu_count):
        self.popup = tk.Toplevel(self.root)
        self.popup.title("Select GPU")
        self.set_window_icon(self.popup)

        label = tk.Label(self.popup, text="Select GPU:")
        label.pack(pady=10)

        self.gpu_var.set(
            torch.cuda.get_device_name(0)
        )  # Default selection is the first GPU
        self.gpu_menu = tk.OptionMenu(
            self.popup,
            self.gpu_var,
            *[torch.cuda.get_device_name(i) for i in range(gpu_count)],
        )
        self.gpu_menu.pack(pady=5)

        confirm_button = tk.Button(
            self.popup, text="Confirm", command=self.confirm_selection
        )
        confirm_button.pack(pady=5)

        cancel_button = tk.Button(
            self.popup, text="Cancel", command=self.cancel_selection
        )
        cancel_button.pack(pady=5)

    def confirm_selection(self):
        selected_gpu = self.gpu_var.get()
        device_id = [
            i
            for i in range(torch.cuda.device_count())
            if torch.cuda.get_device_name(i) == selected_gpu
        ][0]
        torch.cuda.set_device(device_id)
        self.show_popup(f"Using GPU: {selected_gpu}")
        self.popup.destroy()

    def cancel_selection(self):
        self.popup.destroy()

    def show_popup(self, message):
        messagebox.showinfo("GPU Selection", message)

    def set_window_icon(self, window):
        icon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
        if os.path.exists(icon_path):
            window.iconbitmap(icon_path)


class OptionsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("News ML")
        self.set_window_icon()

        self.title_label = tk.Label(
            self.root, text="News ML powered by ABE", font=("Arial", 16, "bold")
        )
        self.cuda_label = None
        self.availability_label = None

        self.buttons = {
            "Run Test": [],
            "Run Training": [],
            "Run Live Analysis": [],
            "Import Data File": [],  # Added the Import Data File button
        }

        # Store the subprocesses for later termination
        self.subprocesses = []

    def create_label(self, text, font=("Arial", 14, "bold"), fg="black"):
        label = tk.Label(self.root, text=text, font=font, fg=fg)
        label.pack(pady=10)
        return label

    def create_button(self, text, width, command):
        button = tk.Button(self.root, text=text, width=width, command=command)
        button.pack(pady=2)
        return button

    def create_button_separator(self):
        button_separator = tk.Frame(self.root, height=1, width=300, bg="gray")
        button_separator.pack(pady=10)

    def on_button_click(self, option, module):
        if option == "Run Test":
            self.run_tests(f"tnt.{module}")
        elif option == "Run Training":
            self.run_train(f"tnt.{module}")
        elif option == "Run Live Analysis":
            self.run_live_analysis()
        elif option == "Import Data File":  # Added the Import Data File option
            self.import_data_file()

    def run_tests(self, module):
        try:
            script_path = os.path.join(
                os.path.dirname(__file__), module.replace(".", "/"), "test.py"
            )
            process = subprocess.Popen(["python", script_path])
            self.subprocesses.append(process)
            self.show_popup("Tests started successfully.")
        except Exception as e:
            self.show_popup(
                f"An error occurred while starting tests:\n\n{str(e)}", error=True
            )

    def run_train(self, module):
        try:
            script_path = os.path.join(
                os.path.dirname(__file__), module.replace(".", "/"), "train.py"
            )
            process = subprocess.Popen(["python", script_path])
            self.subprocesses.append(process)
            self.show_popup("Training started successfully.")
        except Exception as e:
            self.show_popup(
                f"An error occurred while starting training:\n\n{str(e)}", error=True
            )

    def show_popup(self, message, error=False):
        popup = tk.Toplevel(self.root)
        popup.title("Info" if not error else "Error")

        label = tk.Label(popup, text=message)
        label.pack(padx=20, pady=10)

        close_button = tk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=10)

        popup.mainloop()

    def run_all_tests(self):
        try:
            test_script_path = os.path.join(os.path.dirname(__file__), "tnt", "test.py")
            process = subprocess.Popen(["python", test_script_path])
            self.subprocesses.append(process)
            self.show_popup("Tests started successfully.")
        except Exception as e:
            self.show_popup(
                f"An error occurred while starting tests:\n\n{str(e)}", error=True
            )

    def select_gpu(self):
        gpu_selector = GpuSelector(self.root)
        gpu_selector.select_gpu()

    def run_all_training(self):
        try:
            train_script_path = os.path.join(
                os.path.dirname(__file__), "tnt", "train.py"
            )
            process = subprocess.Popen(["python", train_script_path])
            self.subprocesses.append(process)
            self.show_popup("Training started successfully.")
        except Exception as e:
            self.show_popup(
                f"An error occurred while starting training:\n\n{str(e)}", error=True
            )

    def run_live_analysis_button(self):
        loading_bar = LoadingBar(total_stages=0, icon_path="favicon.ico")
        run_with_gui(loading_bar=loading_bar)

    def open_settings(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Settings")
        config_form = ConfigForm(new_window, Config)

    def cancel(self):
        # Terminate all subprocesses
        for process in self.subprocesses:
            process.terminate()
        self.root.destroy()

    def import_data_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Select Data File",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")),
        )
        if file_path:
            destination_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "tnt"
            )
            os.makedirs(destination_dir, exist_ok=True)
            destination_path = os.path.join(destination_dir, "data.csv")

            if os.path.abspath(file_path) != os.path.abspath(destination_path):
                shutil.copy(file_path, destination_path)
                self.show_popup("Data file imported successfully.")
            else:
                self.show_popup("Data file is already in the destination folder.")

    def set_window_icon(self):
        icon_path = os.path.join(os.path.dirname(__file__), "favicon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

    def show_options(self):
        self.set_window_icon()  # Set the window icon

        self.title_label.pack(pady=20)

        # CUDA label
        cuda_status = "Enabled" if torch.cuda.is_available() else "Disabled"
        cuda_color = "green" if torch.cuda.is_available() else "red"
        self.cuda_label = self.create_label(
            f"CUDA: {cuda_status}", font=("Arial", 14, "bold"), fg=cuda_color
        )

        # CUDA availability label
        availability_status = (
            "Available" if torch.cuda.is_available() else "Not Available"
        )
        availability_color = "green" if torch.cuda.is_available() else "red"
        self.availability_label = self.create_label(
            f"Availability: {availability_status}",
            font=("Arial", 12, "bold"),
            fg=availability_color,
        )

        select_gpu_button = self.create_button("Select GPU", 20, self.select_gpu)

        label = self.create_label("Which action would you like to perform?")
        options = ["Run Test", "Run Training"]

        subfolders = ["summary_generation", "sentiment_analysis", "ner"]

        # Run All Tests button
        all_tests_button = self.create_button("Run All Tests", 20, self.run_all_tests)
        all_training_button = self.create_button(
            "Run All Training", 20, self.run_all_training
        )

        for subfolder in subfolders:
            module_label = self.create_label(f"{subfolder.capitalize()}:")
            for option in options:
                button = self.create_button(
                    option,
                    20,
                    lambda option=option, module=subfolder: self.on_button_click(
                        option, module
                    ),
                )
                self.buttons[option].append(button)
            self.create_button_separator()

        live_analysis_button = self.create_button(
            "Run Live Analysis", 20, self.run_live_analysis_button
        )

        settings_button = self.create_button("Settings", 20, self.open_settings)

        import_data_button = self.create_button(
            "Import Data File", 20, self.import_data_file
        )  # Added the Import Data File button

        cancel_button = self.create_button("Exit", 20, self.cancel)

        self.root.mainloop()


class ConfigForm:
    def __init__(self, master, config):
        self.master = master
        self.config = config
        self.entries = {}

        # Create a new scrollable canvas
        canvas = tk.Canvas(master)
        scrollbar = tk.Scrollbar(master, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        # Configure the scroll region of the canvas to fit the scrollable frame
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        # Make the canvas reflect the movements on the scrollbar
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        row = 0
        for var in dir(self.config):
            if not var.startswith("__"):
                label = tk.Label(scrollable_frame, text=var)
                label.grid(row=row, column=0)

                entry = tk.Entry(scrollable_frame)
                entry.insert(0, str(getattr_static(self.config, var)))
                entry.grid(row=row, column=1)

                self.entries[var] = entry
                row += 1

        submit_button = tk.Button(scrollable_frame, text="Submit", command=self.submit)
        submit_button.grid(row=row, column=0, columnspan=2)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def submit(self):
        for var, entry in self.entries.items():
            value = entry.get()
            original_value = getattr(
                self.config, var
            )  # Get the original value from the config
            value_type = type(
                original_value
            ).__name__  # Get the original value's type name

            # Parse the input value according to the original type
            if value_type == "int":
                value = int(value)
            elif value_type == "float":
                value = float(value)
            elif value_type == "list":
                try:
                    value = ast.literal_eval(
                        value
                    )  # Safely evaluate the string as a list
                except (SyntaxError, ValueError):
                    value = original_value  # If parsing fails, use the original value
            elif value_type == "bool":
                value = value.lower() == "true"  # Convert the input value to boolean

            setattr(self.config, var, value)

        # Save the updated configuration to a temporary file
        config_tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config_tmp.py"
        )
        with open(config_tmp_path, "w") as file:
            file.write("import torch\n")
            file.write("from pathlib import Path\n\n")
            file.write("from typing import List\n\n")
            file.write("class Config:\n")
            for var in dir(self.config):
                if not var.startswith("__"):
                    value = getattr(self.config, var)
                    value_type = type(value).__name__
                    if isinstance(value, str):
                        value = value.replace("\\", "\\\\")  # Escape backslashes
                        value = value.replace('"', r"\"")  # Escape double quotes
                        value = f'"{value}"'  # Add double quotes around strings
                    elif isinstance(value, list):
                        value = repr(value)[1:-1]  # Use repr() and remove brackets []
                    else:
                        value = str(value)
                    line = f"    {var}: {value_type} = {value}\n"
                    file.write(line)

        # Backup the original config.py file
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.py"
        )
        config_backup_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config_backup.py"
        )
        shutil.copyfile(config_path, config_backup_path)

        # Replace the original config.py file with the temporary file
        shutil.copyfile(config_tmp_path, config_path)
        os.remove(config_tmp_path)

        print("Configuration updated and written to config.py")
        self.master.destroy()


if __name__ == "__main__":
    options_gui = OptionsGUI()
    options_gui.show_options()
