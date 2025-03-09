# Detailed Setup Guide for VSCode

This guide will walk you through the process of setting up the ColPali + Qwen2-VL Multimodal RAG System in VSCode with a virtual environment.

## Prerequisites

- Python 3.8+ installed
- Git installed
- Visual Studio Code installed
- CUDA-compatible GPU with at least 12GB VRAM (for optimal performance)
- CUDA and cuDNN installed (if using GPU)

## Step 1: Clone the Repository

1. Open a terminal or command prompt
2. Navigate to the directory where you want to clone the repository
3. Run the following command:

```bash
git clone https://github.com/tofunori/colpali-qwen2vl-rag.git
cd colpali-qwen2vl-rag
```

## Step 2: Open the Project in VSCode

1. Open VSCode
2. Choose File > Open Folder... (or use the keyboard shortcut)
3. Select the `colpali-qwen2vl-rag` folder you just cloned
4. VSCode will open the project

## Step 3: Create a Virtual Environment Using VSCode

### Option 1: Using the Command Palette

1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS) to open the Command Palette
2. Type "Python: Create Environment" and select it
3. Choose "Venv" as the environment type
4. Select your Python interpreter (Python 3.8+ recommended)
5. VSCode will create a virtual environment and show a notification when it's done

### Option 2: Using the Terminal in VSCode

1. Open a terminal in VSCode by selecting Terminal > New Terminal from the menu
2. Run one of the following commands depending on your operating system:

   **Windows:**
   ```bash
   python -m venv venv
   ```

   **Linux/macOS:**
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

   **Linux/macOS:**
   ```bash
   source venv/bin/activate
   ```

   After activation, you should see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

## Step 4: Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install all the necessary libraries including:
- torch
- transformers
- byaldi
- pdf2image
- qwen-vl-utils
- flash-attn
- and other dependencies

The installation might take a few minutes depending on your internet connection.

## Step 5: Install Poppler (Required for PDF Processing)

### Windows:

1. Download Poppler for Windows from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
2. Extract the downloaded file to a location on your computer (e.g., `C:\Program Files\poppler`)
3. Add the `bin` directory to your PATH environment variable:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click on "Environment Variables"
   - In the "System variables" section, find the "Path" variable and click "Edit"
   - Click "New" and add the path to the `bin` directory (e.g., `C:\Program Files\poppler\bin`)
   - Click "OK" on all dialogs to save changes

### Linux (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

### macOS:

```bash
brew install poppler
```

## Step 6: Configure VSCode for Python Development

For the best development experience, it's recommended to install the following extensions in VSCode:

1. **Python Extension**: 
   - Click on the Extensions icon in the Activity Bar on the side of VSCode (or press `Ctrl+Shift+X`)
   - Search for "Python"
   - Install the extension by Microsoft

2. **Pylance** (for better code intelligence):
   - Search for "Pylance" in the Extensions view
   - Install the extension by Microsoft

3. **Jupyter** (optional, for notebook support):
   - Search for "Jupyter" in the Extensions view
   - Install the "Jupyter" extension by Microsoft

Once these extensions are installed, VSCode will provide features like IntelliSense, linting, debugging, and more for Python development.

## Step 7: Test the Installation

To test if the installation was successful:

1. Make sure your virtual environment is activated
2. Create a simple PDF file for testing or use an existing one
3. Run a simple command to index a document and ask a question:

```bash
# From the terminal in VSCode
python cli.py index --files path/to/your/document.pdf
python cli.py ask "What is this document about?"
```

## Step 8: Set Up a Launch Configuration (Optional)

If you want to debug the application, you can set up a launch configuration in VSCode:

1. Click on the Run and Debug icon in the Activity Bar (or press `Ctrl+Shift+D`)
2. Click on "create a launch.json file"
3. Select "Python" as the environment
4. Choose "Python File" as the configuration
5. Edit the created `launch.json` file to include configurations for both indexing and asking:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Index Documents",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/cli.py",
            "args": ["index", "--folder", "${input:folderPath}"],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Ask Question",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/cli.py",
            "args": ["ask", "${input:question}"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ],
    "inputs": [
        {
            "id": "folderPath",
            "type": "promptString",
            "description": "Path to folder containing PDFs",
            "default": "./sample_pdfs"
        },
        {
            "id": "question",
            "type": "promptString",
            "description": "Question to ask about the documents",
            "default": "What is this document about?"
        }
    ]
}
```

## Step 9: Troubleshooting Common Issues

### Issue: ImportError related to torch or flash-attn

**Solution:**
- Make sure you have the correct CUDA version installed for your GPU
- Try reinstalling PyTorch with the specific CUDA version you have:
  ```bash
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cu118 # Replace cu118 with your CUDA version
  ```

### Issue: Missing DLLs on Windows

**Solution:**
- Make sure you've added the Poppler bin directory to your PATH
- You might need to restart VSCode or your computer after changing environment variables

### Issue: Out of Memory errors

**Solution:**
- Try the low memory mode: `python cli.py ask --low-memory "Your question here"`
- Reduce the number of pages retrieved: `python cli.py ask --top-k 1 "Your question here"`
- Close other applications that use GPU memory

### Issue: PDF processing errors

**Solution:**
- Make sure Poppler is correctly installed and in your PATH
- Try a different PDF file to see if it's a specific file issue

## Step 10: Creating a Sample PDF for Testing

If you don't have a PDF with both text and images for testing, you can create one using this simple Python script:

```python
from fpdf import FPDF
import os

# Create a PDF with text and an image
def create_sample_pdf(output_path="sample.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Sample Document for Testing", 0, 1, "C")
    
    # Add text
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(190, 10, "This is a sample document with text and an image for testing the ColPali + Qwen2-VL Multimodal RAG System. This system can understand both text and visual content in documents.")
    
    # Add more text
    pdf.ln(10)
    pdf.multi_cell(190, 10, "The system uses ColPali for document retrieval and Qwen2-VL for understanding the content and answering questions.")
    
    # Create a simple image
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a new image with white background
        img = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(img)
        
        # Draw a rectangle
        draw.rectangle([(50, 50), (350, 150)], outline="black")
        
        # Draw text
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((100, 100), "This is a test image", fill="black", font=font)
        
        # Save the image
        img_path = "temp_image.png"
        img.save(img_path)
        
        # Add the image to PDF
        pdf.image(img_path, x=10, y=100, w=180)
        
        # Remove the temporary image
        os.remove(img_path)
        
    except ImportError:
        pdf.cell(190, 10, "Could not create image. PIL library required.", 0, 1)
    
    # Add more text after the image
    pdf.ln(110)
    pdf.multi_cell(190, 10, "When you ask questions about this document, the system should be able to reference both the text content and the visual elements.")
    
    # Save the PDF
    pdf.output(output_path)
    print(f"Sample PDF created at: {output_path}")

# Install FPDF if needed
try:
    import fpdf
except ImportError:
    import pip
    pip.main(["install", "fpdf"])
    
# Create the sample PDF
create_sample_pdf()
```

Save this script as `create_sample.py` and run it to generate a simple test PDF.

## Conclusion

You now have a fully functional multimodal RAG system set up in VSCode! You can use it to:

1. Index PDF documents that contain both text and images
2. Ask questions about the content of these documents
3. Get answers that take into account both textual and visual information

If you encounter any issues or have suggestions for improvements, feel free to contribute to the repository by opening an issue or submitting a pull request.

Happy querying!
