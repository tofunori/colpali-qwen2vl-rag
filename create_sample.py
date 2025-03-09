#!/usr/bin/env python3
"""
Script to create a sample PDF document with text and images for testing the RAG system.
"""

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
        
        # Add a bar chart
        draw.rectangle([(75, 130), (100, 150)], fill="blue")
        draw.rectangle([(125, 110), (150, 150)], fill="blue")
        draw.rectangle([(175, 90), (200, 150)], fill="blue")
        draw.rectangle([(225, 70), (250, 150)], fill="blue")
        draw.rectangle([(275, 130), (300, 150)], fill="blue")
        
        # Add labels
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((75, 155), "A", fill="black", font=font)
        draw.text((125, 155), "B", fill="black", font=font)
        draw.text((175, 155), "C", fill="black", font=font)
        draw.text((225, 155), "D", fill="black", font=font)
        draw.text((275, 155), "E", fill="black", font=font)
        
        # Add title
        draw.text((150, 50), "Figure 1: Sample Chart", fill="black", font=font)
        
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
    pdf.set_font("Arial", "B", 14)
    pdf.cell(190, 10, "Key Information", 0, 1)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(190, 10, "When you ask questions about this document, the system should be able to reference both the text content and the visual elements.")
    
    # Add a table
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 10, "Item", 1, 0, "C")
    pdf.cell(40, 10, "Value", 1, 0, "C")
    pdf.cell(90, 10, "Description", 1, 1, "C")
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(60, 10, "Alpha", 1, 0)
    pdf.cell(40, 10, "23.4", 1, 0, "R")
    pdf.cell(90, 10, "First parameter", 1, 1)
    
    pdf.cell(60, 10, "Beta", 1, 0)
    pdf.cell(40, 10, "45.6", 1, 0, "R")
    pdf.cell(90, 10, "Second parameter", 1, 1)
    
    pdf.cell(60, 10, "Gamma", 1, 0)
    pdf.cell(40, 10, "78.9", 1, 0, "R")
    pdf.cell(90, 10, "Third parameter", 1, 1)
    
    # Add conclusion
    pdf.ln(10)
    pdf.multi_cell(190, 10, "According to Figure 1, item C and D have the highest values. The table shows that Gamma has the highest value at 78.9.")
    
    # Save the PDF
    pdf.output(output_path)
    print(f"Sample PDF created at: {output_path}")


if __name__ == "__main__":
    # Check if PIL is installed, if not try to install it
    try:
        import PIL
    except ImportError:
        print("PIL not found. Attempting to install...")
        import pip
        pip.main(["install", "pillow"])
    
    # Check if FPDF is installed, if not try to install it
    try:
        import fpdf
    except ImportError:
        print("FPDF not found. Attempting to install...")
        import pip
        pip.main(["install", "fpdf"])
    
    # Create sample directory if needed
    os.makedirs("sample_pdfs", exist_ok=True)
    
    # Create the sample PDF
    create_sample_pdf("sample_pdfs/sample.pdf")
    
    print("\nNow you can test the RAG system with:")
    print("python cli.py index --folder ./sample_pdfs")
    print("python cli.py ask \"What does Figure 1 show?\"")
    print("python cli.py ask \"What is the value of Gamma in the table?\"")
