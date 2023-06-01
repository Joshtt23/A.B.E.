import tkinter as tk
from tkinter import messagebox
import importlib
import torch
from config import Config


class GpuSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU Selection")
        self.gpu_var = tk.StringVar(self.root)
        self.gpu_menu = None

    def select_gpu(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                self.create_gpu_selection_box(gpu_count)
            else:
                messagebox.showwarning("GPU Selection", "No GPUs available.")
        else:
            messagebox.showwarning("GPU Selection", "CUDA is not available.")

    def create_gpu_selection_box(self, gpu_count):
        label = tk.Label(self.root, text="Select GPU:")
        label.pack(pady=10)

        self.gpu_var.set(
            torch.cuda.get_device_name(0)
        )  # Default selection is the first GPU
        self.gpu_menu = tk.OptionMenu(
            self.root,
            self.gpu_var,
            *[torch.cuda.get_device_name(i) for i in range(gpu_count)],
        )
        self.gpu_menu.pack(pady=5)

        confirm_button = tk.Button(
            self.root, text="Confirm", command=self.confirm_selection
        )
        confirm_button.pack(pady=5)

    def confirm_selection(self):
        selected_gpu = self.gpu_var.get()
        device_id = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ].index(selected_gpu)
        torch.cuda.set_device(device_id)
        messagebox.showinfo("GPU Selection", f"Using GPU: {selected_gpu}")
        self.root.destroy()

    def run(self):
        self.root.mainloop()


class SettingsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Settings")
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        self.canvas = tk.Canvas(self.frame)
        self.scrollbar = tk.Scrollbar(
            self.frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def create_label_entry(self, text, default_value):
        label = tk.Label(self.scrollable_frame, text=text)
        label.pack(pady=5)
        entry_var = tk.StringVar(self.scrollable_frame, value=str(default_value))
        entry = tk.Entry(self.scrollable_frame, textvariable=entry_var, width=30)
        entry.pack(pady=5)
        return entry_var

    def save_settings(self):
        for option, entry_var in self.config_entries.items():
            setattr(Config, option, entry_var.get())
        messagebox.showinfo("Settings", "Settings saved successfully.")
        self.root.destroy()

    def run(self):
        self.config_entries = {}

        num_columns = max(len(Config.__dict__) // 2, 1)
        grid_options = {"sticky": "w", "padx": 5, "pady": 5}

        i = 0
        for option, default_value in Config.__dict__.items():
            if not option.startswith("__"):
                entry_var = self.create_label_entry(option, default_value)
                self.config_entries[option] = entry_var
                i += 1

        save_button = tk.Button(
            self.scrollable_frame, text="Save", command=self.save_settings
        )
        save_button.pack(pady=5)

        self.root.mainloop()


class OptionsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("News ML")

        self.title_label = tk.Label(
            self.root, text="News ML powered by ABE", font=("Arial", 16, "bold")
        )
        self.cuda_label = None
        self.availability_label = None

        self.buttons = {"Run Test": [], "Run Training": [], "Run Live Analysis": []}

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
            self.run_tests(module)
        elif option == "Run Training":
            self.run_train(module)
        elif option == "Run Live Analysis":
            self.run_live_analysis()

    def run_tests(self, module):
        test_module = f"tnt.{module}.test"
        try:
            test = importlib.import_module(test_module)
            test.run_tests()
        except ImportError:
            messagebox.showerror("Error", f"Cannot import test module for {module}")

    def run_train(self, module):
        train_module = f"tnt.{module}.train"
        try:
            train = importlib.import_module(train_module)
            train.run_training()
        except ImportError:
            messagebox.showerror("Error", f"Cannot import train module for {module}")

    def show_options(self):
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

        subfolders = ["summary_generator", "sentiment_analysis", "keyword_extraction"]
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
            "Run Live Analysis",
            20,
            lambda: self.on_button_click("Run Live Analysis", ""),
        )

        # Run All Tests button
        all_tests_button = self.create_button("Run All Tests", 20, self.run_all_tests)

        # Run All Training button
        all_training_button = self.create_button(
            "Run All Training", 20, self.run_all_training
        )

        settings_button = self.create_button("Settings", 20, self.open_settings)

        cancel_button = self.create_button("Exit", 20, self.cancel)

        self.root.mainloop()

    def select_gpu(self):
        gpu_selector = GpuSelector()
        gpu_selector.run()

    def run_live_analysis(self):
        import main

        main.run_live_analysis()

    def run_all_tests(self):
        import tnt.test as test

        test.run_tests()

    def run_all_training(self):
        import tnt.train as train

        train.run_training()

    def open_settings(self):
        settings_gui = SettingsGUI()
        settings_gui.run()

    def cancel(self):
        self.root.destroy()


if __name__ == "__main__":
    options_gui = OptionsGUI()
    options_gui.show_options()
