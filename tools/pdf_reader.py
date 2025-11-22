"""
PDF Reader Module for extracting raw content from PDF files.
This module handles PDF file reading and text extraction.
"""

import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFReader:
    """
    A class to handle PDF file reading and text extraction.
    Supports multiple extraction methods for better compatibility.
    """

    def __init__(self):
        self.supported_formats = [".pdf"]

    def validate_pdf_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid PDF file.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            bool: True if valid PDF, False otherwise
        """
        try:
            file_path = Path(file_path)

            # Check if file exists
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False

            # Check file extension
            if file_path.suffix.lower() not in self.supported_formats:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False

            # Try to open the file to validate it's a proper PDF
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                # Just check if we can access basic info
                _ = len(pdf_reader.pages)

            return True

        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False

    def extract_text_pypdf2(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF using PyPDF2.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            text_content = ""

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"

            return text_content.strip()

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            return None

    def extract_text_pdfplumber(self, file_path: str) -> Optional[str]:
        """
        Extract text from PDF using pdfplumber (more accurate for complex layouts).

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            text_content = ""

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"

            return text_content.strip()

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return None

    def extract_raw_content(self, file_path: str) -> Dict[str, Any]:
        """
        Extract raw content from PDF file using multiple methods for best results.

        Args:
            file_path (str): Path to the PDF file

        Returns:
            Dict[str, Any]: Dictionary containing extraction results and metadata
        """
        result = {
            "success": False,
            "file_path": file_path,
            "text_content": "",
            "extraction_method": None,
            "page_count": 0,
            "error_message": None,
        }

        try:
            # Validate PDF file first
            if not self.validate_pdf_file(file_path):
                result["error_message"] = "Invalid PDF file"
                return result

            # Get page count
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                result["page_count"] = len(pdf_reader.pages)

            # Try pdfplumber first (usually more accurate)
            text_content = self.extract_text_pdfplumber(file_path)
            if text_content and text_content.strip():
                result["text_content"] = text_content
                result["extraction_method"] = "pdfplumber"
                result["success"] = True
                logger.info(
                    f"Successfully extracted text using pdfplumber from {file_path}"
                )
                return result

            # Fallback to PyPDF2
            text_content = self.extract_text_pypdf2(file_path)
            if text_content and text_content.strip():
                result["text_content"] = text_content
                result["extraction_method"] = "PyPDF2"
                result["success"] = True
                logger.info(
                    f"Successfully extracted text using PyPDF2 from {file_path}"
                )
                return result

            # If both methods fail or return empty content
            result["error_message"] = "No text content could be extracted from the PDF"
            logger.warning(f"No text content extracted from {file_path}")

        except Exception as e:
            result["error_message"] = f"Extraction failed: {str(e)}"
            logger.error(f"PDF extraction failed for {file_path}: {str(e)}")

        return result


def read_pdf(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract raw content from a PDF file.

    Args:
        file_path (str): Path to the PDF file

    Returns:
        Dict[str, Any]: Extraction results
    """
    reader = PDFReader()
    return reader.extract_raw_content(file_path)
