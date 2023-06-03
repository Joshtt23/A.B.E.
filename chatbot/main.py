import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
from transformers import pipeline
from config import Config


class TextGenerator:
    def __init__(self, config):
        self.config = config
        self.generator = pipeline("text-generation", model=self.config.GENERATION_MODEL)

    def generate_text(self, prompt):
        try:
            generated_text = self.generator(
                prompt,
                max_length=self.config.MAX_SUMMARY_LENGTH,
                min_length=self.config.MIN_SUMMARY_LENGTH,
                num_return_sequences=1,
            )[0]["generated_text"]
            return generated_text
        except Exception as e:
            messagebox.showerror(
                "Error", f"An error occurred while generating text:\n\n{str(e)}"
            )


class TextGeneratorGUI:
    def __init__(self, root, generator):
        self.root = root
        self.generator = generator
        self.text_prompt = None
        self.text_output = None

    def create_label(self, text):
        label = tk.Label(self.root, text=text)
        label.pack(pady=10)

    def create_text_input(self):
        self.text_prompt = scrolledtext.ScrolledText(self.root, width=60, height=10)
        self.text_prompt.pack(pady=10)

    def create_text_output(self):
        self.text_output = scrolledtext.ScrolledText(self.root, width=60, height=10)
        self.text_output.pack(pady=10)

    def generate_button_click(self):
        prompt = self.text_prompt.get("1.0", tk.END).strip()
        if prompt:
            generated_text = self.generator.generate_text(prompt)
            self.text_output.delete("1.0", tk.END)
            self.text_output.insert(tk.END, generated_text)
        else:
            messagebox.showwarning("Warning", "Please enter a prompt.")

    def create_generate_button(self):
        generate_button = tk.Button(
            self.root, text="Generate", command=self.generate_button_click
        )
        generate_button.pack(pady=10)

    def run(self):
        self.create_label("Text Generation")
        self.create_text_input()
        self.create_generate_button()
        self.create_text_output()
        self.root.mainloop()


if __name__ == "__main__":
    config = Config()

    root = tk.Tk()
    root.title("Text Generator")
    root.geometry("400x400")

    generator = TextGenerator(config)

    gui = TextGeneratorGUI(root, generator)
    gui.run()
