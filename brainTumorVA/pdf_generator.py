# pdf_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os

# Output directory
OUTPUT_DIR = "outputs/pdf_reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_pdf(original_img_path, segmented_img_path, llm_response, patient_data):
    """
    Generate a hospital-grade, professionally formatted medical report PDF.
    Includes full error handling and graceful fallbacks.
    """
    # Output path
    filename = f"report_{int(datetime.now().timestamp())}.pdf"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            leftMargin=0.7 * inch,
            rightMargin=0.7 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )
        styles = getSampleStyleSheet()
        story = []

        # ================================
        # HEADER: Hospital-style Header
        # ================================
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.darkblue,
            alignment=TA_CENTER,
            spaceAfter=6
        )
        story.append(Paragraph("AI NEUROIMAGING ANALYSIS REPORT", header_style))
        story.append(Paragraph("Department of Radiology & AI Diagnostics", styles['Italic']))
        story.append(Spacer(1, 8))

        # Metadata
        meta_style = ParagraphStyle('Meta', fontSize=10, textColor=colors.grey)
        story.append(Paragraph(
            f"<i>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System: BrainTumorVA v1.0</i>",
            meta_style
        ))
        story.append(Spacer(1, 12))

        # ================================
        # PATIENT INFO SECTION
        # ================================
        story.append(Paragraph("📋 PATIENT INFORMATION", styles['Heading3']))
        story.append(Spacer(1, 6))

        try:
            patient_lines = [line.strip() for line in patient_data.split('\n') if line.strip()]
            data = []
            for line in patient_lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    data.append([
                        Paragraph(f"<b>{parts[0].strip()}</b>", styles['Normal']),
                        Paragraph(parts[1].strip(), styles['Normal'])
                    ])

            if data:
                patient_table = Table(data, colWidths=[150, 350])
                patient_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('LINEBELOW', (1, 0), (1, -1), 0.5, colors.lightgrey),
                    ('PADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(patient_table)
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"<i>⚠️ Could not display patient data: {str(e)}</i>", styles['Italic']))
            story.append(Spacer(1, 12))

        # ================================
        # AI CLINICAL INTERPRETATION
        # ================================
        story.append(Paragraph("📄 AI CLINICAL INTERPRETATION", styles['Heading3']))
        story.append(Spacer(1, 6))

        try:
            formatted_text = ""
            for line in llm_response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("**") and line.endswith("**"):
                    formatted_text += f"<br/><b>{line[2:-2]}</b><br/>"
                elif line.startswith("#"):
                    continue
                else:
                    formatted_text += f"{line}<br/>"

            llm_style = ParagraphStyle(
                'LLMStyle',
                parent=styles['Normal'],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
                spaceAfter=10
            )
            story.append(Paragraph(formatted_text, llm_style))
        except Exception as e:
            story.append(Paragraph(f"<i>❌ Error formatting clinical report: {str(e)}</i>", styles['Italic']))
        story.append(Spacer(1, 12))

        # ================================
        # IMAGE COMPARISON (Aligned Pair)
        # ================================
        story.append(Paragraph("🖼️ IMAGING FINDINGS", styles['Heading3']))
        story.append(Spacer(1, 8))

        img_width = 4 * inch
        img_height = 3 * inch

        try:
            # Validate image paths
            if not os.path.exists(original_img_path):
                raise FileNotFoundError(f"Original image not found: {original_img_path}")
            if not os.path.exists(segmented_img_path):
                raise FileNotFoundError(f"Segmented image not found: {segmented_img_path}")

            # Load images
            original_img = Image(original_img_path, width=img_width, height=img_height)
            segmented_img = Image(segmented_img_path, width=img_width, height=img_height)

            # Image grid
            img_data = [[original_img, segmented_img]]
            img_table = Table(img_data, colWidths=[img_width, img_width])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            # Caption
            caption_table = Table(
                [["Original MRI", "Tumor Highlighted"]],
                colWidths=[img_width, img_width]
            )
            caption_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.darkslategray),
            ]))

            story.append(img_table)
            story.append(caption_table)

        except Exception as img_error:
            story.append(Paragraph(
                f"<i style='color:red;'>📷 Image Error: {str(img_error)}. "
                "The images could not be embedded in the report.</i>",
                styles['Normal']
            ))

        story.append(Spacer(1, 16))

        # ================================
        # FOOTER
        # ================================
        footer_style = ParagraphStyle(
            'Footer',
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER,
            spaceBefore=10
        )
        story.append(Paragraph(
            "This report was generated by an AI system for assistance only. "
            "Always consult a qualified radiologist for final diagnosis.",
            footer_style
        ))

        # ================================
        # BUILD PDF
        # ================================
        try:
            doc.build(story)
            print(f"✅ Professional PDF report generated: {output_path}")
            return output_path
        except Exception as build_error:
            print(f"❌ Failed to build PDF: {build_error}")
            return None

    except Exception as e:
        print(f"❌ Unexpected error in PDF generation: {e}")
        return None
    