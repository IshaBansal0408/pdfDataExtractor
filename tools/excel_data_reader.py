"""
Excel Data Reader Module
This module reads Excel files to understand data structure and format.
Used to analyze the expected output format from Excel files.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelDataReader:
    """
    A class to read and analyze Excel file structure and content.
    Helps understand the expected output format for data extraction.
    """

    def __init__(self):
        self.supported_formats = [".xlsx", ".xls", ".csv"]

    def validate_excel_file(self, file_path: str) -> bool:
        """
        Validate if the file is a valid Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            bool: True if valid Excel file, False otherwise
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

            return True

        except Exception as e:
            logger.error(f"Excel validation failed: {str(e)}")
            return False

    def read_excel_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Read Excel file and analyze its structure.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            Dict[str, Any]: Structure analysis of the Excel file
        """
        result = {
            "success": False,
            "file_path": file_path,
            "sheets": [],
            "total_sheets": 0,
            "structure": {},
            "sample_data": {},
            "error_message": None,
        }

        try:
            if not self.validate_excel_file(file_path):
                result["error_message"] = "Invalid Excel file"
                return result

            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            result["total_sheets"] = len(excel_file.sheet_names)
            result["sheets"] = excel_file.sheet_names

            # Analyze each sheet
            for sheet_name in excel_file.sheet_names:
                logger.info(f"Analyzing sheet: {sheet_name}")

                # Read sheet data
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Analyze structure
                sheet_structure = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "non_null_counts": df.count().to_dict(),
                    "null_counts": df.isnull().sum().to_dict(),
                }

                # Get sample data (first 5 rows)
                sample_data = df.head().fillna("").to_dict("records")

                result["structure"][sheet_name] = sheet_structure
                result["sample_data"][sheet_name] = sample_data

            result["success"] = True
            logger.info(f"Successfully analyzed Excel file: {file_path}")

        except Exception as e:
            result["error_message"] = f"Failed to read Excel file: {str(e)}"
            logger.error(f"Excel reading failed: {str(e)}")

        return result

    def extract_data_schema(self, file_path: str) -> Dict[str, Any]:
        """
        Extract a data schema from Excel file that can be used as a template.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            Dict[str, Any]: Data schema for JSON output structure
        """
        schema = {
            "success": False,
            "schema": {},
            "field_definitions": {},
            "error_message": None,
        }

        try:
            structure_result = self.read_excel_structure(file_path)

            if not structure_result["success"]:
                schema["error_message"] = structure_result["error_message"]
                return schema

            # Create schema for each sheet
            for sheet_name, sheet_structure in structure_result["structure"].items():
                sheet_schema = {}
                field_definitions = {}

                for column_name in sheet_structure["column_names"]:
                    # Determine field type based on data type
                    data_type = sheet_structure["data_types"].get(column_name, "object")

                    if "int" in data_type:
                        field_type = "integer"
                    elif "float" in data_type:
                        field_type = "number"
                    elif "datetime" in data_type:
                        field_type = "datetime"
                    elif "bool" in data_type:
                        field_type = "boolean"
                    else:
                        field_type = "string"

                    sheet_schema[column_name] = field_type

                    # Add field definition with metadata
                    field_definitions[column_name] = {
                        "type": field_type,
                        "required": sheet_structure["null_counts"].get(column_name, 0)
                        == 0,
                        "description": f"Field for {column_name.lower().replace('_', ' ')}",
                    }

                schema["schema"][sheet_name] = sheet_schema
                schema["field_definitions"][sheet_name] = field_definitions

            schema["success"] = True

        except Exception as e:
            schema["error_message"] = f"Failed to extract schema: {str(e)}"
            logger.error(f"Schema extraction failed: {str(e)}")

        return schema

    def generate_sample_json(
        self, file_path: str, sheet_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a sample JSON structure based on Excel data.

        Args:
            file_path (str): Path to the Excel file
            sheet_name (str, optional): Specific sheet to process. If None, processes first sheet.

        Returns:
            Dict[str, Any]: Sample JSON structure
        """
        result = {
            "success": False,
            "sample_json": {},
            "json_template": {},
            "error_message": None,
        }

        try:
            structure_result = self.read_excel_structure(file_path)

            if not structure_result["success"]:
                result["error_message"] = structure_result["error_message"]
                return result

            # Use specified sheet or first available sheet
            if sheet_name is None:
                sheet_name = structure_result["sheets"][0]
            elif sheet_name not in structure_result["sheets"]:
                result["error_message"] = f"Sheet '{sheet_name}' not found"
                return result

            sample_data = structure_result["sample_data"][sheet_name]

            # Create JSON template (empty structure)
            if sample_data:
                json_template = {key: "" for key in sample_data[0].keys()}
                result["json_template"] = json_template

                # Create sample JSON with actual data
                result["sample_json"] = sample_data

            result["success"] = True

        except Exception as e:
            result["error_message"] = f"Failed to generate JSON: {str(e)}"
            logger.error(f"JSON generation failed: {str(e)}")

        return result

    def analyze_expected_output(self, file_path: str) -> Dict[str, Any]:
        """
        Complete analysis of expected output Excel file.

        Args:
            file_path (str): Path to the Expected Output.xlsx file

        Returns:
            Dict[str, Any]: Complete analysis including structure, schema, and samples
        """
        analysis = {
            "success": False,
            "file_analysis": {},
            "data_schema": {},
            "sample_json": {},
            "extraction_template": {},
            "error_message": None,
        }

        try:
            # Get file structure
            structure_result = self.read_excel_structure(file_path)
            if not structure_result["success"]:
                analysis["error_message"] = structure_result["error_message"]
                return analysis

            analysis["file_analysis"] = structure_result

            # Get data schema
            schema_result = self.extract_data_schema(file_path)
            if schema_result["success"]:
                analysis["data_schema"] = schema_result

            # Generate sample JSON
            json_result = self.generate_sample_json(file_path)
            if json_result["success"]:
                analysis["sample_json"] = json_result

                # Create extraction template for LLM
                analysis["extraction_template"] = {
                    "format": "json",
                    "structure": json_result["json_template"],
                    "instructions": "Extract data from PDF text and format according to this JSON structure",
                }

            analysis["success"] = True
            logger.info(f"Successfully analyzed expected output file: {file_path}")

        except Exception as e:
            analysis["error_message"] = f"Analysis failed: {str(e)}"
            logger.error(f"Expected output analysis failed: {str(e)}")

        return analysis


def analyze_expected_output(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to analyze expected output Excel file.

    Args:
        file_path (str): Path to the Excel file

    Returns:
        Dict[str, Any]: Analysis results
    """
    reader = ExcelDataReader()
    return reader.analyze_expected_output(file_path)
