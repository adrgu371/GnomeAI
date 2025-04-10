#!/usr/bin/env python3
# Gnome AI with Qwen2.5 and AI_Best.py features, no MLX, for Linux x86, single chat
#
# Copyright (C) 2025 [Your Name]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
import os
import platform
import aiohttp
import asyncio
import warnings
import logging
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from cachetools import TTLCache
import json
from datetime import datetime
import customtkinter as ctk
from tkinter import scrolledtext, filedialog, messagebox, font, Menu
from PIL import Image
import re
import threading
import time
import PyPDF2
from docx import Document
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
CONFIG_FILE = "config.json"
try:
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
        BRAVE_API_KEY = config.get("brave_api_key", None)
except (FileNotFoundError, json.JSONDecodeError):
    BRAVE_API_KEY = None
    config = {}

CHAT_HISTORY_FILE = "conversation_history.json"
TEMPERATURE = 0.3
TOP_P = 0.5
MAX_TOKENS = 1500
cache = TTLCache(maxsize=100, ttl=3600)
CURRENT_DATE = "April 06, 2025"

logging.basicConfig(filename='gnome_ai.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def load_conversation():
    default_structure = []
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            content = file.read().strip()
            if not content:
                return default_structure
            try:
                data = json.loads(content)
                return data if isinstance(data, list) else default_structure
            except json.JSONDecodeError:
                return default_structure
    return default_structure

def save_conversation(chat_data):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(chat_data, file, indent=4)

async def search_brave_async(query):
    if not BRAVE_API_KEY:
        return "Web search unavailable without a Brave API key."
    if query in cache:
        return cache[query]
    async with aiohttp.ClientSession() as session:
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": BRAVE_API_KEY}
            params = {"q": query, "count": 5}
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    error_msg = f"Brave Search failed: {response.status}"
                    cache[query] = error_msg
                    return error_msg
                data = await response.json()
                results = data.get("web", {}).get("results", [])
                if not results:
                    result = "No relevant information found."
                    cache[query] = result
                    return result
                combined = "\n".join([f"Source: {r['url']}\n{r.get('description', 'No content')}" for r in results[:3]])
                cache[query] = combined
                return combined
        except Exception as e:
            error_msg = f"Error with Brave Search: {str(e)}"
            cache[query] = error_msg
            return error_msg

async def fetch_x_post(post_url):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(post_url) as response:
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                content = soup.get_text(strip=True)[:500]
                return content if content else "No content extracted."
        except Exception as e:
            return f"Error fetching X post: {str(e)}"

def get_ollama_models():
    try:
        result = os.popen("ollama list").read()
        lines = result.strip().splitlines()[1:]
        return [line.split()[0].strip() for line in lines] or ["llama3.2:3b"]
    except Exception:
        return ["llama3.2:3b"]

def initialize_llm():
    try:
        llm = OllamaLLM(model="llama3.2:3b", temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, base_url="http://localhost:11434")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLaMA3.2: {str(e)}")
        raise

def generate_response(llm, query, history, internet_data=None, x_data=None):
    prompt = (
        f"Current Date: {CURRENT_DATE}\n"
        f"You are an AI assistant built by xAI. Answer using available data.\n"
        f"Conversation History: {' '.join([f'User: {h["user"]} AI: {h["gnome"]}' for h in history[-3:]]) if history else 'None'}\n"
        f"Internet Data: {internet_data or 'None'}\n"
        f"X Data: {x_data or 'None'}\n"
        f"Query: {query}\n"
        "Provide a detailed, factual response. If image generation is requested, ask for confirmation. "
        "If asked who deserves to die, say you’re not allowed to make that choice. Use web/X data if provided."
    )
    try:
        response = llm.invoke(prompt).replace(prompt, "").strip()
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

async def process_query(llm, query, history, use_internet=False):
    force_search = False
    if "search" in query.lower():
        force_search = True
        search_term = query.lower().replace("search", "").replace("please", "").strip()
    elif "who is" in query.lower() or re.match(r".* (is|was) [a-zA-Z\s]+$", query.lower()):
        force_search = True
        search_term = query.lower().replace("who is", "").replace("is", "").replace("was", "").strip()

    if (force_search or "search the web" in query.lower()) and use_internet and BRAVE_API_KEY:
        if not force_search:
            search_term = query.replace("search the web", "").strip()
        internet_data = await search_brave_async(search_term)
        response = generate_response(llm, query, history, internet_data)
    elif "x.com" in query.lower() and use_internet:
        post_url = re.search(r'(https?://x\.com/[^\s]+)', query)
        if post_url:
            x_data = await fetch_x_post(post_url.group(0))
            response = generate_response(llm, query, history, x_data=x_data)
        else:
            response = generate_response(llm, query, history)
    elif "generate an image" in query.lower():
        response = "Do you want me to generate an image based on this description? Please confirm."
    elif "deserves to die" in query.lower() or "deserves the death penalty" in query.lower():
        response = "As an AI, I’m not allowed to make that choice."
    else:
        if use_internet and BRAVE_API_KEY:
            internet_data = await search_brave_async(query)
            response = generate_response(llm, query, history, internet_data)
        else:
            response = generate_response(llm, query, history)
    return response

def read_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() for page in reader.pages)
    elif file_path.endswith((".doc", ".docx")):
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    elif file_path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(file_path)
        return df.to_string(index=False)
    return "Unsupported file type."

class ChatOnlyAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gnome AI")
        self.geometry("1200x600")
        self.minsize(800, 400)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.stop_typing = False
        self.use_internet = False
        self.text_size = 14
        self.chat_font = font.Font(family="TkDefaultFont", size=self.text_size)
        self.monitor_font = font.Font(family="TkDefaultFont", size=14)
        self.chat_history = load_conversation()
        self.is_loading = False

        # Main layout
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Content frame
        self.content_frame = ctk.CTkFrame(self.main_frame)
        self.content_frame.pack(fill="both", expand=True)

        self.notebook = ctk.CTkTabview(self.content_frame)
        self.notebook.pack(fill="both", expand=True)

        self.chat_tab = self.notebook.add("Chat")
        self.chat_display = scrolledtext.ScrolledText(self.chat_tab, wrap="word", height=15, font=self.chat_font)
        self.chat_display.pack(padx=5, pady=5, fill="both", expand=True)
        self.chat_display.configure(state="disabled")
        self.setup_chat_display()
        self.load_initial_chat()

        self.brave_api_frame = ctk.CTkFrame(self.content_frame)
        self.brave_api_frame.pack(padx=10, pady=5, fill="x")
        self.brave_api_button = ctk.CTkButton(self.brave_api_frame, text="Set Brave API Key", command=self.set_brave_api_key)
        self.brave_api_button.pack(side="top")

        self.monitor_tab = self.notebook.add("Monitoring")
        self.monitor_display = scrolledtext.ScrolledText(self.monitor_tab, wrap="word", height=15, state="disabled", font=self.monitor_font)
        self.monitor_display.pack(padx=5, pady=5, fill="both", expand=True)
        self.update_monitor_display()

        # Input frame
        self.input_frame = ctk.CTkFrame(self.content_frame)
        self.input_frame.pack(padx=10, pady=10, fill="x", expand=False)

        self.models = get_ollama_models()
        self.selected_model = ctk.StringVar(value="llama3.2:3b" if "llama3.2:3b" in self.models else self.models[0])
        self.model_menu = ctk.CTkOptionMenu(self.input_frame, values=self.models, variable=self.selected_model, command=self.change_model, width=120)
        self.model_menu.pack(padx=5, pady=5, side="left")

        self.tools_menu_button = ctk.CTkButton(self.input_frame, text="Tools", command=self.show_tools_menu)
        self.tools_menu_button.pack(padx=5, pady=5, side="left")

        self.input_field = ctk.CTkEntry(self.input_frame, placeholder_text="Ask Gnome anything...")
        self.input_field.pack(padx=5, pady=5, side="left", fill="x", expand=True)
        self.setup_input_field()

        self.button_frame = ctk.CTkFrame(self.input_frame)
        self.button_frame.pack(padx=5, pady=5, side="right")

        self.submit_button = ctk.CTkButton(self.button_frame, text="Send", command=self.submit_query)
        self.submit_button.pack(padx=2, pady=2, side="left")
        self.upload_files_button = ctk.CTkButton(self.button_frame, text="Upload Files", command=self.upload_files)
        self.upload_files_button.pack(padx=2, pady=2, side="left")
        self.toggle_internet_button = ctk.CTkButton(self.button_frame, text="Toggle Internet", command=self.toggle_internet_mode)
        self.toggle_internet_button.pack(padx=2, pady=2, side="left")

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_tasks, daemon=True)
        self.thread.start()

        self.llm = initialize_llm()
        self.use_history = True

        # Bind Ctrl + mouse wheel for text size adjustment
        self.bind_all("<Control-MouseWheel>", self.adjust_text_size)
        self.bind_all("<Control-Button-4>", self.adjust_text_size)  # For Linux
        self.bind_all("<Control-Button-5>", self.adjust_text_size)  # For Linux

        self.after(200, self.on_resize)
        self.type_text("Gnome: Ready to assist with LLaMA3.2!\n\n")
        self.autoresize_buttons()

    def on_closing(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.destroy()

    def setup_chat_display(self):
        self.chat_menu = Menu(self.chat_display, tearoff=0)
        self.chat_menu.add_command(label="Copy", command=self.copy_chat_text)
        if platform.system() == "Darwin":
            self.chat_display.bind("<Button-2>", self.show_chat_menu)
            self.chat_display.bind("<Control-Button-1>", self.show_chat_menu)
        else:
            self.chat_display.bind("<Button-3>", self.show_chat_menu)

    def setup_input_field(self):
        self.input_menu = Menu(self.input_field, tearoff=0)
        self.input_menu.add_command(label="Copy", command=self.copy_input_text)
        self.input_menu.add_command(label="Paste", command=self.paste_input_text)
        if platform.system() == "Darwin":
            self.input_field.bind("<Button-2>", self.show_input_menu)
            self.input_field.bind("<Control-Button-1>", self.show_input_menu)
        else:
            self.input_field.bind("<Button-3>", self.show_input_menu)

    def show_chat_menu(self, event):
        self.chat_menu.tk_popup(event.x_root, event.y_root)

    def show_input_menu(self, event):
        self.input_menu.tk_popup(event.x_root, event.y_root)

    def copy_chat_text(self):
        try:
            selected_text = self.chat_display.selection_get()
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except:
            pass

    def copy_input_text(self):
        try:
            selected_text = self.input_field.get()
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except:
            pass

    def paste_input_text(self):
        try:
            clipboard_text = self.clipboard_get()
            self.input_field.delete(0, "end")
            self.input_field.insert(0, clipboard_text)
        except:
            pass

    def load_initial_chat(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete(1.0, "end")
        for entry in self.chat_history:
            self.chat_display.insert("end", f"You: {entry['user']}\nGnome: {entry['gnome']}\n\n")
        self.chat_display.configure(state="disabled")

    def show_tools_menu(self):
        menu = Menu(self.tools_menu_button, tearoff=0)
        menu.add_command(label="Clear Context", command=self.clear_context)
        menu.add_command(label="Stop", command=self.stop_response)
        menu.add_command(label="Clear Chat", command=self.clear_chat_display)
        menu.tk_popup(self.tools_menu_button.winfo_rootx(), 
                     self.tools_menu_button.winfo_rooty() + self.tools_menu_button.winfo_height())

    def adjust_text_size(self, event):
        if event.delta > 0 or event.num == 4:  # Scroll up or Button-4 (Linux)
            self.text_size = min(20, self.text_size + 1)
        elif event.delta < 0 or event.num == 5:  # Scroll down or Button-5 (Linux)
            self.text_size = max(8, self.text_size - 1)
        self.chat_font.configure(size=self.text_size)
        self.chat_display.configure(font=self.chat_font)

    def on_resize(self, event=None):
        window_width = self.winfo_width()
        chat_height = max(10, (self.winfo_height() - 200) // 30)
        self.chat_display.configure(height=chat_height)
        self.monitor_display.configure(height=chat_height)
        input_field_width = max(300, window_width - 600)
        self.input_frame.pack_configure(padx=5, pady=5, fill="x", expand=False)
        self.input_field.configure(width=input_field_width)
        self.autoresize_buttons()

    def autoresize_buttons(self):
        window_width = self.winfo_width()
        input_field_width = self.input_field.winfo_width()
        left_side_width = self.model_menu.winfo_width() + self.tools_menu_button.winfo_width() + 30
        available_width = window_width - input_field_width - left_side_width - 40

        buttons = [
            self.submit_button,
            self.upload_files_button,
            self.toggle_internet_button,
        ]
        num_buttons = len(buttons)

        if num_buttons == 0 or available_width <= 0:
            return

        button_width = max(50, min(100, available_width // num_buttons - 10))
        for button in buttons:
            button.configure(width=button_width)

    def set_brave_api_key(self):
        global BRAVE_API_KEY, config
        dialog = ctk.CTkInputDialog(title="Set Brave API Key", text="Enter your Brave API Key:")
        entry = dialog.entry  # Access the entry widget from CTkInputDialog

        # Add right-click menu for copy-paste
        api_menu = Menu(entry, tearoff=0)
        api_menu.add_command(label="Copy", command=lambda: self.copy_api_key(entry))
        api_menu.add_command(label="Paste", command=lambda: self.paste_api_key(entry))
        
        if platform.system() == "Darwin":
            entry.bind("<Button-2>", lambda event: api_menu.tk_popup(event.x_root, event.y_root))
            entry.bind("<Control-Button-1>", lambda event: api_menu.tk_popup(event.x_root, event.y_root))
        else:
            entry.bind("<Button-3>", lambda event: api_menu.tk_popup(event.x_root, event.y_root))

        api_key = dialog.get_input()
        if api_key:
            BRAVE_API_KEY = api_key.strip()
            config["brave_api_key"] = BRAVE_API_KEY
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            self.type_text("Brave API Key successfully added.\n\n")

    def copy_api_key(self, entry):
        try:
            selected_text = entry.get()
            self.clipboard_clear()
            self.clipboard_append(selected_text)
        except:
            pass

    def paste_api_key(self, entry):
        try:
            clipboard_text = self.clipboard_get()
            entry.delete(0, "end")
            entry.insert(0, clipboard_text)
        except:
            pass

    def stop_response(self):
        self.stop_typing = True
        self.is_loading = False
        self.type_text("Gnome: Response stopped.\n\n")

    def clear_context(self):
        self.use_history = False
        self.chat_history.clear()
        self.chat_display.configure(state="normal")
        self.chat_display.delete(1.0, "end")
        self.chat_display.configure(state="disabled")
        self.type_text("Gnome: Context and chat history cleared. Starting fresh.\n\n")
        save_conversation(self.chat_history)

    def clear_chat_display(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete(1.0, "end")
        self.chat_display.configure(state="disabled")
        self.type_text("Gnome: Chat display cleared.\n\n")

    def toggle_internet_mode(self):
        self.use_internet = not self.use_internet
        mode_status = "enabled" if self.use_internet else "disabled"
        self.type_text(f"Gnome: Internet mode {mode_status}.\n\n")

    def upload_files(self):
        file_path = filedialog.askopenfilename(filetypes=[
            ("All Supported Files", "*.png *.jpg *.jpeg *.gif *.txt *.pdf *.doc *.docx *.xls *.xlsx"),
            ("Image Files", "*.png *.jpg *.jpeg *.gif"),
            ("Text Files", "*.txt"),
            ("PDF Files", "*.pdf"),
            ("Word Documents", "*.doc *.docx"),
            ("Excel Files", "*.xls *.xlsx")
        ])
        if file_path:
            if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
                self.process_image(file_path)
            else:
                self.process_file(file_path)

    def process_file(self, file_path):
        content = read_file(file_path)
        if "Unsupported" in content:
            self.type_text(f"Gnome: {content}\n\n")
        else:
            response = generate_response(self.llm, f"Analyze this file content: {content[:1000]}", self.chat_history)
            self.type_text(f"Gnome: {response}\n\n")
            self.chat_history.append({"user": f"Uploaded file: {os.path.basename(file_path)}", "gnome": response, "timestamp": datetime.now().isoformat()})
            save_conversation(self.chat_history)

    def process_image(self, file_path):
        file_name = os.path.basename(file_path)
        self.type_text(f"Gnome: Uploaded image '{file_name}'. Please describe it for analysis, or I’ll assume it’s a generic image.\n\n")
        response = generate_response(self.llm, f"Describe a generic image based on the file name '{file_name}'", self.chat_history)
        self.type_text(f"Gnome: {response}\n\n")
        self.chat_history.append({"user": f"Uploaded image: {file_name}", "gnome": response, "timestamp": datetime.now().isoformat()})
        save_conversation(self.chat_history)

    def type_text(self, text, delay=0.02):
        self.stop_typing = False
        self.chat_display.configure(state="normal")
        current_text = self.chat_display.get("1.0", "end-1c")
        if current_text and not current_text.endswith("\n"):
            self.chat_display.insert("end", "\n")
        for char in text:
            if self.stop_typing:
                break
            self.chat_display.insert("end", char)
            self.chat_display.see("end")
            self.update()
            time.sleep(delay)
        self.chat_display.insert("end", "\n")
        self.chat_display.configure(state="disabled")

    def show_loading_animation(self):
        if self.is_loading:
            self.chat_display.configure(state="normal")
            self.chat_display.delete("end-2c", "end-1c")  # Remove the previous symbol
            current_symbol = self.chat_display.get("end-2c", "end-1c") or "|"
            next_symbol = {"|": "/", "/": "-", "-": "\\", "\\": "|"}.get(current_symbol, "|")
            self.chat_display.insert("end", next_symbol)
            self.chat_display.see("end")
            self.chat_display.configure(state="disabled")
            self.after(100, self.show_loading_animation)

    def start_loading(self):
        self.is_loading = True
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", "Gnome: Processing... |")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")
        self.show_loading_animation()

    def stop_loading(self):
        self.is_loading = False
        self.chat_display.configure(state="normal")
        self.chat_display.delete("end-14c", "end")  # Remove "Processing... |"
        self.chat_display.configure(state="disabled")

    def update_monitor_display(self):
        try:
            with open('gnome_ai.log', 'r') as log_file:
                logs = log_file.readlines()
                self.monitor_display.configure(state="normal")
                self.monitor_display.delete(1.0, "end")
                self.monitor_display.insert("end", "".join(logs))
                self.monitor_display.see("end")
                self.monitor_display.configure(state="disabled")
        except FileNotFoundError:
            self.monitor_display.configure(state="normal")
            self.monitor_display.delete(1.0, "end")
            self.monitor_display.insert("end", "Log file not found.\n")
            self.monitor_display.configure(state="disabled")
        self.after(5000, self.update_monitor_display)

    def run_async_tasks(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def schedule_task(self, coro):
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    def change_model(self, model_name):
        try:
            self.llm = OllamaLLM(model=model_name, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS, base_url="http://localhost:11434")
            self.type_text(f"Gnome: Switched to {model_name}.\n\n")
        except Exception as e:
            self.type_text(f"Gnome: Failed to switch model: {str(e)}\n\n")

    def submit_query(self):
        query = self.input_field.get().strip()
        if query:
            self.chat_display.configure(state="normal")
            self.chat_display.insert("end", f"You: {query}\n")
            self.chat_display.configure(state="disabled")
            self.input_field.delete(0, "end")
            self.start_loading()
            self.schedule_task(self.process_query(query))

    async def process_query(self, query):
        response = await process_query(self.llm, query, self.chat_history, self.use_internet)
        self.stop_loading()
        self.type_text(f"Gnome: {response}\n\n")
        self.chat_history.append({"user": query, "gnome": response, "timestamp": datetime.now().isoformat()})
        save_conversation(self.chat_history)

if __name__ == "__main__":
    app = ChatOnlyAIApp()
    app.mainloop()
