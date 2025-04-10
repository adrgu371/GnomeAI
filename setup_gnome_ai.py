#!/usr/bin/env python3
# Setup script for Gnome AI to install dependencies and Ollama
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

import os
import sys
import platform
import subprocess
import urllib.request
import shutil
import time

# Define the required Python dependencies for gnome_ai.py
REQUIRED_PACKAGES = [
    "aiohttp",
    "beautifulsoup4",
    "langchain-ollama",
    "cachetools",
    "customtkinter",
    "Pillow",
    "PyPDF2",
    "python-docx",
    "pandas",
    "openpyxl",
]

# Define paths and URLs
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_LINUX_URL = "https://ollama.com/download/ollama-linux-amd64"
OLLAMA_MAC_URL = "https://ollama.com/download/ollama-darwin"
OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"

def run_command(command, shell=False, check=True):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, shell=shell, check=check, text=True, capture_output=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error running command {command}: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error running command {command}: {str(e)}")
        sys.exit(1)

def check_python_version():
    """Ensure Python 3.6 or higher is installed."""
    if sys.version_info < (3, 6):
        print("Python 3.6 or higher is required. Please install a newer version of Python.")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected.")

def install_pip_dependencies():
    """Install required Python packages using pip."""
    print("Checking and installing Python dependencies...")
    # Ensure pip is installed and up-to-date
    run_command([sys.executable, "-m", "ensurepip", "--upgrade"])
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    for package in REQUIRED_PACKAGES:
        print(f"Installing {package}...")
        stdout, stderr = run_command([sys.executable, "-m", "pip", "install", package])
        if stdout:
            print(stdout)
        if stderr and "already satisfied" not in stderr.lower():
            print(f"Warning: {stderr}")

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        stdout, stderr = run_command(["ollama", "--version"], check=False)
        if "version" in stdout.lower():
            print("Ollama is already installed.")
            return True
        return False
    except FileNotFoundError:
        return False

def install_ollama_windows():
    """Install Ollama on Windows."""
    print("Downloading Ollama for Windows...")
    ollama_exe = "OllamaSetup.exe"
    urllib.request.urlretrieve(OLLAMA_WINDOWS_URL, ollama_exe)
    
    print("Installing Ollama... Please follow the installer prompts.")
    subprocess.run(ollama_exe, shell=True)
    
    # Wait for the user to complete the installation
    input("Press Enter after the Ollama installation is complete...")
    
    # Clean up
    if os.path.exists(ollama_exe):
        os.remove(ollama_exe)
    
    # Verify installation
    if check_ollama_installed():
        print("Ollama installed successfully on Windows.")
    else:
        print("Ollama installation failed. Please install it manually from https://ollama.com/download")
        sys.exit(1)

def install_ollama_linux():
    """Install Ollama on Linux."""
    print("Installing Ollama on Linux...")
    # Download the official install script and run it
    stdout, stderr = run_command(["curl", "-fsSL", "https://ollama.com/install.sh"], check=False)
    with open("install_ollama.sh", "w") as f:
        f.write(stdout)
    
    # Make the script executable and run it
    run_command(["chmod", "+x", "install_ollama.sh"])
    run_command(["./install_ollama.sh"], shell=True)
    
    # Clean up
    if os.path.exists("install_ollama.sh"):
        os.remove("install_ollama.sh")
    
    # Verify installation
    if check_ollama_installed():
        print("Ollama installed successfully on Linux.")
    else:
        print("Ollama installation failed. Please install it manually from https://ollama.com/download")
        sys.exit(1)

def install_ollama_macos():
    """Install Ollama on macOS."""
    print("Downloading Ollama for macOS...")
    ollama_zip = "ollama-darwin.zip"
    urllib.request.urlretrieve(OLLAMA_MAC_URL, ollama_zip)
    
    print("Extracting and installing Ollama...")
    shutil.unpack_archive(ollama_zip, "ollama-darwin")
    
    # Move the binary to /usr/local/bin
    run_command(["sudo", "mv", "ollama-darwin/ollama", "/usr/local/bin/ollama"])
    
    # Clean up
    shutil.rmtree("ollama-darwin")
    if os.path.exists(ollama_zip):
        os.remove(ollama_zip)
    
    # Verify installation
    if check_ollama_installed():
        print("Ollama installed successfully on macOS.")
    else:
        print("Ollama installation failed. Please install it manually from https://ollama.com/download")
        sys.exit(1)

def install_ollama():
    """Install Ollama based on the operating system."""
    if check_ollama_installed():
        return
    
    os_name = platform.system().lower()
    if os_name == "windows":
        install_ollama_windows()
    elif os_name == "linux":
        install_ollama_linux()
    elif os_name == "darwin":
        install_ollama_macos()
    else:
        print(f"Unsupported operating system: {os_name}. Please install Ollama manually from https://ollama.com/download")
        sys.exit(1)

def start_ollama_service():
    """Start the Ollama service."""
    print("Starting Ollama service...")
    try:
        # Run ollama serve in the background
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Give it a moment to start
        print("Ollama service started.")
    except Exception as e:
        print(f"Failed to start Ollama service: {str(e)}")
        print("Please start it manually by running 'ollama serve' in a terminal.")
        sys.exit(1)

def pull_ollama_model():
    """Pull the specified Ollama model."""
    print(f"Checking for {OLLAMA_MODEL} model...")
    stdout, stderr = run_command(["ollama", "list"], check=False)
    if OLLAMA_MODEL in stdout:
        print(f"{OLLAMA_MODEL} is already installed.")
        return
    
    print(f"Pulling {OLLAMA_MODEL} model...")
    stdout, stderr = run_command(["ollama", "pull", OLLAMA_MODEL])
    if stdout:
        print(stdout)
    if stderr:
        print(f"Warning: {stderr}")
    
    # Verify the model was pulled
    stdout, stderr = run_command(["ollama", "list"])
    if OLLAMA_MODEL in stdout:
        print(f"{OLLAMA_MODEL} model pulled successfully.")
    else:
        print(f"Failed to pull {OLLAMA_MODEL}. Please try pulling it manually with 'ollama pull {OLLAMA_MODEL}'.")
        sys.exit(1)

def main():
    print("Setting up Gnome AI environment...")
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Install Python dependencies
    install_pip_dependencies()
    
    # Step 3: Install Ollama
    install_ollama()
    
    # Step 4: Start Ollama service
    start_ollama_service()
    
    # Step 5: Pull the required model
    pull_ollama_model()
    
    print("Setup complete! You can now run the Gnome AI script with 'python3 gnome_ai.py'.")

if __name__ == "__main__":
    main()
