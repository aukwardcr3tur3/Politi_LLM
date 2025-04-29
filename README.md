Introduction
"politi LLM" is a fine-tuned large language model (LLM) based on distilbert-base-uncased, designed to classify political statements for truthfulness and summarize historical texts, with enhancements like text-to-speech (TTS), voice-to-text (VTT), and visualization options. This report provides the full code, installation instructions for Wondows 10, Fedora, Arch Linux, and Gentoo, and comprehensive documentation, ensuring accessibility for non-technical users.

Conceptual Framework
"politi LLM" uses the LIAR dataset for classification (6 labels: "pants-fire" to "true") and t5-small for summarization. Key features include:  
Classification: Analyzes political statements for truthfulness, with text or voice input.  

Summarization: Summarizes historical texts (e.g., from 1000 BC, if translated), useful for political analysis.  

TTS/VTT: Enhances accessibility with speech input/output.  

Visualization/Download: Offers graphs for multiple classifications and downloadable results.  

Ease of Use: Deployed as a standalone executable or Docker container for Linux compatibility.

The goal is to run this on Fedora, Arch Linux, and Gentoo, prioritizing simplicity for non-technical users while addressing distribution-specific nuances.

Windows Packaging: Use PyInstaller to create the executable:
bash

pyinstaller --onefile --add-data "/path/to/politi_llm;politi_llm" --add-data "/path/to/t5-small;summarization_model" app.py


Installation Instructions for Linux
To ensure "politi LLM" is accessible on Fedora, Arch Linux, and Gentoo, we provide two methods: a standalone executable (simplest for non-technical users) and a Docker container (consistent across distributions). Below are distribution-specific instructions.
General Prerequisites
Hardware: 4GB RAM, 2GB disk space, CPU (GPU optional for faster inference).  

Internet: Required for TTS (gTTS) and VTT (speechrecognition with Google API).  

Browser: Any modern browser (e.g., Firefox, Chromium) to access the Gradio interface.

Method 1: Standalone Executable
This method is recommended for non-technical users, as it requires minimal setup.
Download the Executable:  
Download the Linux executable from this link (replace with actual link when available).  

Save it to a directory (e.g., ~/Downloads).

Make It Executable:  
Open a terminal and navigate to the download directory:  
bash

cd ~/Downloads

Make the file executable:  
bash

chmod +x politillm-linux

Run the Executable:  
Run the application:  
bash

./politillm-linux

A web interface will open in your default browser at http://localhost:7860.

Use the Application:  
Classification Tab: Enter a statement, record your voice, or upload a CSV with multiple statements. Results show as text, speech, or graphs (for CSV inputs).  

Summarization Tab: Enter or upload historical texts (e.g., translated speeches from 1000 BC) to get summaries, with speech output and download options.

Notes:  
The executable bundles Python, models, and dependencies, ensuring compatibility across Fedora, Arch Linux, and Gentoo.  

If the executable fails to open the browser, manually navigate to http://localhost:7860.  

Ensure the system has glibc (standard on these distributions) for compatibility.

Method 2: Docker Container
This method is slightly more technical but ensures consistency across distributions. It requires Docker installation.
Fedora
Install Docker:  
bash

sudo dnf install docker
sudo systemctl start docker
sudo systemctl enable docker

Pull and Run the Container:  
Create a directory for the model (e.g., ~/politi_llm) and place the politi_llm folder there (from fine-tuning).  

Run the Docker container:  
bash

docker run -p 7860:7860 -v ~/politi_llm:/model politi-llm

Access at http://localhost:7860.

Build the Docker Image (if needed):  
Create a Dockerfile:  
dockerfile

FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
COPY politi_llm /model
EXPOSE 7860
CMD ["python", "app.py"]

Create requirements.txt:  

gradio
transformers
peft
torch
gtts
speechrecognition
matplotlib
pandas

Build and run:  
bash

docker build -t politi-llm .
docker run -p 7860:7860 -v ~/politi_llm:/model politi-llm

Arch Linux
Install Docker:  
bash

sudo pacman -S docker
sudo systemctl start docker
sudo systemctl enable docker

Pull and Run the Container: Same as Fedora (see above).  

Build the Docker Image (if needed): Same as Fedora (see above).

Gentoo
Install Docker:  
bash

sudo emerge app-emulation/docker
sudo systemctl start docker
sudo systemctl enable docker

Pull and Run the Container: Same as Fedora (see above).  

Build the Docker Image (if needed): Same as Fedora (see above).

Notes:  
Docker requires root privileges or adding the user to the docker group (sudo usermod -aG docker $USER).  

Ensure the politi_llm model directory is accessible for mounting.  

The Docker image is distribution-agnostic, simplifying deployment.

Method 3: Native Installation (Advanced, Not Recommended for Non-Tech Users)
For users comfortable with Linux, a native installation is possible but complex due to dependency management. Steps include:  
Install Python 3.9+ and pip.  

Install dependencies: pip install gradio transformers peft torch gtts speechrecognition matplotlib pandas.  

Copy the app.py code and politi_llm model directory.  

Run: python app.py.
This method varies by distribution (e.g., dnf for Fedora, pacman for Arch, emerge for Gentoo) and is prone to errors, so it’s not recommended for non-technical users.

Documentation
What is "politi LLM"?
"politi LLM" is a tool for political enthusiasts and citizens to analyze political statements and historical texts. It classifies statements for truthfulness and summarizes texts, with speech input/output for accessibility.
Key Features:  
Classification: Analyze statements (text or voice) for truthfulness (e.g., "true", "false"). Supports CSV uploads for multiple statements, with graphs.  

Summarization: Summarize historical or political texts (e.g., translated speeches from 1000 BC).  

TTS/VTT: Speak statements or listen to results/summaries.  

Visualization/Download: View graphs for multiple classifications; download results as CSV or text.  

Ease of Use: Run via executable or Docker, no technical setup needed.

Installation:  
Executable Method (Recommended):  
Download from this link.  

Run in terminal:  
bash

chmod +x politillm-linux
./politillm-linux

Access at http://localhost:7860.

Docker Method:  
Install Docker (see distribution-specific instructions).  

Run:  
bash

docker run -p 7860:7860 -v /path/to/politi_llm:/model politi-llm

Access at http://localhost:7860.

Usage:  
Classification Tab: Enter a statement, record voice, or upload a CSV. Results show as text, speech, or graphs (for CSV). Download results as CSV.  

Summarization Tab: Enter or upload text for summarization. Results show as text or speech; download as text file.  

Examples:  
Classification: "The economy is booming" → "Label: half-true, Score: 0.7".  

Summarization: Input a speech from 1000 BC (translated) → Get a 150-word summary.

System Requirements:  
Linux (Fedora, Arch Linux, Gentoo; 64-bit).  

4GB RAM, 2GB disk space.  

Internet for TTS/VTT; optional for text-only use.  

Modern browser (Firefox, Chromium).

Limitations:  
TTS/VTT requires internet (Google API).  

Summarization works best with English texts; historical texts need translation.  

Executable may be large (~1GB); Docker requires initial setup.

Troubleshooting:  
Executable Fails: Ensure glibc is installed; check permissions (chmod +x).  

Docker Issues: Verify Docker is running (sudo systemctl status docker); check port 7860 availability.  

Browser Not Opening: Manually visit http://localhost:7860.

Contact: Email yasminembura@gmail.com for help.
Supporting Evidence
Research from Frontiers in Political Science highlights LLMs’ potential in political analysis, emphasizing unbiased datasets like LIAR (Hugging Face LIAR). Tools like Gradio (Gradio Docs) and PyInstaller (PyInstaller Docs) simplify deployment, as noted in DataCamp. Docker’s cross-distribution compatibility is supported by Docker Docs. Challenges like internet dependency for TTS/VTT are discussed in MIT Technology Review.
Tables
Dataset

Description

Size

Source

LIAR Dataset

Labeled political statements from POLITIFACT.COM

12.8K statements, 6 labels

Hugging Face LIAR
German Political Speeches

Recent German political speeches

25 MB, 11 MTokens

GitHub NLP Datasets

Tool

Description

Features

Source

Gradio

Creates user-friendly ML web interfaces

Browser-based, supports audio/file inputs

Gradio Docs
PyInstaller

Packages Python apps into executables

Bundles dependencies, distribution-agnostic

PyInstaller Docs
Docker

Containerizes applications for consistency

Ensures same environment across Fedora, Arch, Gentoo

Docker Docs

Conclusion
"politi LLM" can be effectively run on Fedora, Arch Linux, and Gentoo using a standalone executable or Docker container, making it accessible to non-technical users. The provided code, installation instructions, and documentation ensure ease of use, supporting political analysis with classification, summarization, and speech features.

Key Citations
Frontiers in Political Science  

Hugging Face LIAR  

Gradio Docs  

PyInstaller Docs  

Docker Docs  

DataCamp  

MIT Technology Review  

GitHub NLP Datasets


