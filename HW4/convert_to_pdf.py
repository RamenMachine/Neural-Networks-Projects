#!/usr/bin/env python3
"""
Convert VSCode notebook files to PDF format
Extracts content from executed notebooks and generates PDF reports
"""

import os
import re
import base64
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkgreen
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def read_notebook_file(filepath):
    """Read VSCode notebook file and extract content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_cells(content):
    """Extract cells from VSCode notebook format"""
    cells = []
    
    # Pattern to match VSCode cells
    cell_pattern = r'<VSCode\.Cell id="([^"]*)" language="([^"]*)">\n(.*?)\n</VSCode\.Cell>'
    
    matches = re.finditer(cell_pattern, content, re.DOTALL)
    
    for match in matches:
        cell_id = match.group(1)
        language = match.group(2)
        cell_content = match.group(3)
        
        cells.append({
            'id': cell_id,
            'language': language,
            'content': cell_content
        })
    
    return cells

def create_pdf_report(notebook_path, output_path, title):
    """Create PDF report from notebook content"""
    
    # Read notebook content
    content = read_notebook_file(notebook_path)
    cells = extract_cells(content)
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        textColor=darkgreen
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=10,
        spaceAfter=12,
        leftIndent=20,
        backgroundColor='#f0f0f0'
    )
    
    # Build story
    story = []
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 20))
    
    # Process cells
    for i, cell in enumerate(cells):
        if cell['language'] == 'markdown':
            # Process markdown content
            lines = cell['content'].split('\n')
            for line in lines:
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], heading_style))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], styles['Heading2']))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], styles['Heading3']))
                elif line.strip():
                    story.append(Paragraph(line, styles['Normal']))
                story.append(Spacer(1, 6))
        
        elif cell['language'] == 'python':
            # Add code block
            story.append(Paragraph("Python Code:", heading_style))
            
            # Clean up code content
            code_lines = cell['content'].split('\n')
            code_text = '\n'.join(code_lines)
            
            # Add code as preformatted text
            story.append(Paragraph(f"<pre>{code_text}</pre>", code_style))
            story.append(Spacer(1, 12))
    
    # Add results summary
    results_dir = f"results_{notebook_path.split('_')[0].lower()}"
    if os.path.exists(results_dir):
        story.append(Paragraph("Generated Results:", heading_style))
        story.append(Paragraph(f"All analysis results and visualizations have been generated and saved in the '{results_dir}' directory.", styles['Normal']))
        
        # List generated files
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            story.append(Paragraph(f"Generated {len(files)} files including plots and analysis data.", styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated: {output_path}")

def main():
    """Main function to convert both notebooks to PDF"""
    
    # Install required package
    import subprocess
    import sys
    
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])
        import reportlab
    
    # Convert Lab 1
    if os.path.exists('Lab1_Report.ipynb'):
        create_pdf_report('Lab1_Report.ipynb', 'Lab1_Report.pdf', 'Lab 1 Report - ECE 491: 1-D Convolution Analysis')
        print("✓ Lab 1 PDF generated successfully")
    
    # Convert Lab 2  
    if os.path.exists('Lab2_Report.ipynb'):
        create_pdf_report('Lab2_Report.ipynb', 'Lab2_Report.pdf', 'Lab 2 Report - ECE 491: 2-D Image Processing Analysis')
        print("✓ Lab 2 PDF generated successfully")
    
    print("\n📄 PDF Generation Complete!")
    print("=" * 50)
    print("Both notebooks have been converted to PDF format with:")
    print("• Complete notebook content and code")
    print("• Proper formatting and structure")
    print("• References to generated results and visualizations")
    print("• Professional document layout")

if __name__ == "__main__":
    main()