# Gnome AI

Gnome AI is a cross-platform AI chat application built with Python, utilizing the LLaMA3.2 model via Ollama for local inference. It supports web searches using the Brave Search API, file uploads (text, PDFs, Word documents, Excel files), and a user-friendly GUI powered by CustomTkinter. The application is designed to run on Windows, Linux, and macOS, providing a seamless experience for users interested in interacting with a local AI assistant.

## Features

- **Local AI Chat**: Powered by LLaMA3.2 (via Ollama) for privacy-focused, on-device inference or any model you want to use.
- **Web Search Integration**: Use the Brave Search API to fetch real-time web data (requires an API key).
- **File Upload Support**: Upload and analyze text files, PDFs, Word documents, and Excel files.
- **Cross-Platform**: Compatible with Windows, Linux, and macOS.
- **User-Friendly GUI**: Built with CustomTkinter, featuring a chat interface, monitoring tab, and adjustable text size.
- **Loading Animation**: Displays a rotating line animation while processing queries.
- **Copy-Paste Support**: Right-click to copy and paste in the input field and Brave API key dialog.
- **Conversation History**: Saves chat history to a JSON file for persistence across sessions.


## Installation

### Prerequisites

- **Python 3.6 or higher**: Ensure Python is installed on your system.
- **Internet Connection**: Required for downloading dependencies and Ollama.
- **Administrative Privileges**: May be required for installing Ollama on Linux/macOS.


```bash
### Step 1: Clone the Repository
git clone https://github.com/[YourGitHubUsername]/gnome-ai.git
cd gnome-ai

Step 2: Run the Setup Script

On Windows
python setup_gnome_ai.py

Linux/macOS:
chmod +x setup_gnome_ai.py
./setup_gnome_ai.py

### The script will:

Check for Python 3.6+.
Install required Python packages (aiohttp, beautifulsoup4, langchain-ollama, etc.).
Install Ollama if not already present.
Start the Ollama service.
Pull the llama3.2:3b model

Step 3: Run Gnome AI
After setup, start the application:
python3 gnome_ai.py

Usage
Launch the Application:
Run python3 gnome_ai.py to start Gnome AI.
The GUI will open with a chat tab and a monitoring tab.
Interact with Gnome AI:
Type your query in the input field and click "Send".
Use the "Tools" menu to clear context, stop responses, or clear the chat display.
Toggle internet mode to enable/disable web searches (requires a Brave API key).
Set Brave API Key:
Click "Set Brave API Key" to enable web searches.
Right-click in the input dialog to copy/paste your API key.
Obtain a Brave API key from Brave Search API.
Upload Files:
Click "Upload Files" to upload text, PDFs, Word documents, or Excel files for analysis.
Supported file types: .txt, .pdf, .doc, .docx, .xls, .xlsx.
Adjust Text Size:
Use Ctrl + Mouse Wheel to increase or decrease the chat text size.
