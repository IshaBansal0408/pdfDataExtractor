"""
Excel Generator Module
This module generates Excel files from structured JSON data.
Creates output files matching the expected format defined in Expected Output.xlsx
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelGenerator:
    """
    Generate Excel files from structured data extracted by LLM.
    Creates output matching the format specified in Expected Output.xlsx
    """

    def __init__(self):
        self.output_template = {
            "#": "row_number",
            "Key": "field_name",
            "Value": "field_value",
            "Comments": "additional_info",
            "": "",  # Empty column
        }

        # Initialize field mappings and comments dictionaries
        self.field_mappings = {}
        self.field_comments = {}

        # Field info will be loaded lazily when needed
        self._field_info_loaded = False

    def _load_field_info(self):
        """Load field mappings and comments from Expected Output.xlsx directly"""
        try:
            # Load field mappings and comments directly from Expected Output.xlsx
            import pandas as pd

            # Read Expected Output.xlsx directly
            df = pd.read_excel(
                "output/Expected Output.xlsx", header=1
            )  # Skip first row, use second as headers

            # Create mappings from the Excel file
            for _, row in df.iterrows():
                if pd.notna(row.get("Key")) and pd.notna(row.get("Value")):
                    key = str(row["Key"]).strip()
                    value = str(row["Value"]).strip()
                    comment = (
                        str(row.get("Comments", ""))
                        if pd.notna(row.get("Comments"))
                        else ""
                    )

                    # Convert display name to field key format
                    field_key = key.lower().replace(" ", "_").replace("-", "_")

                    self.field_mappings[field_key] = key
                    self.field_comments[field_key] = comment

        except Exception as e:
            logger.warning(f"Could not load field info from Expected Output.xlsx: {e}")
            # Fallback mappings
            self.field_mappings = {
                "first_name": "First Name",
                "last_name": "Last Name",
                "date_of_birth": "Date of Birth",
                "birth_city": "Birth City",
                "birth_state": "Birth State",
                "age": "Age",
                "blood_group": "Blood Group",
                "nationality": "Nationality",
                "joining_date_of_first_professional_role": "Joining Date of First Professional Role",
                "designation_of_first_professional_role": "Designation of First Professional Role",
                "salary_of_first_professional_role": "Salary of First Professional Role",
                "salary_currency_of_first_professional_role": "Salary Currency of First Professional Role",
                "current_organization": "Current Organization",
                "current_joining_date": "Current Joining Date",
                "current_designation": "Current Designation",
                "current_salary": "Current Salary",
                "current_salary_currency": "Current Salary Currency",
                "previous_organization": "Previous Organization",
                "previous_joining_date": "Previous Joining Date",
                "previous_end_year": "Previous End Year",
                "previous_starting_designation": "Previous Starting Designation",
                "high_school": "High School",
                "12th_standard_pass_out_year": "12th Standard Pass Out Year",
                "12th_overall_board_score": "12th Overall Board Score",
                "undergraduate_degree": "Undergraduate Degree",
                "undergraduate_college": "Undergraduate College",
                "undergraduate_year": "Undergraduate Year",
                "undergraduate_cgpa": "Undergraduate CGPA",
                "graduation_degree": "Graduation Degree",
                "graduation_college": "Graduation College",
                "graduation_year": "Graduation Year",
                "graduation_cgpa": "Graduation CGPA",
                "certifications_1": "Certifications 1",
                "certifications_2": "Certifications 2",
                "certifications_3": "Certifications 3",
                "certifications_4": "Certifications 4",
            }
            self.field_comments = {}

    def create_excel_data(
        self, structured_data: Dict[str, Any], metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Convert structured JSON data to Excel-ready format with proper Key-Value-Comments structure.

        Args:
            structured_data (Dict[str, Any]): Extracted structured data
            metadata (Optional[Dict]): Additional metadata to include

        Returns:
            List[Dict]: Data ready for Excel conversion
        """
        # Load field info if not already loaded
        if not self._field_info_loaded:
            self._load_field_info()
            self._field_info_loaded = True

        excel_rows = []

        # Add single header row only
        excel_rows.append(
            {"#": "#", "Key": "Key", "Value": "Value", "Comments": "Comments", "": ""}
        )

        row_number = 1

        # Add each field from structured data (skip metadata fields)
        for json_key, value in structured_data.items():
            if json_key.startswith("_"):  # Skip metadata fields
                continue

            display_name = self.field_mappings.get(
                json_key, json_key.replace("_", " ").title()
            )

            # Get comment for this field
            comment = self.field_comments.get(json_key, "")

            # Format value properly
            formatted_value = str(value) if value else ""

            excel_row = {
                "#": row_number,
                "Key": display_name,
                "Value": formatted_value,
                "Comments": comment,
                "": "",
            }
            excel_rows.append(excel_row)
            row_number += 1

        return excel_rows

    def generate_excel_file(
        self,
        structured_data: Dict[str, Any],
        output_path: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate Excel file from structured data.

        Args:
            structured_data (Dict[str, Any]): Extracted data to convert
            output_path (str): Path where Excel file should be saved
            metadata (Optional[Dict]): Additional metadata to include

        Returns:
            Dict[str, Any]: Generation results with success status
        """
        result = {
            "success": False,
            "output_file": output_path,
            "rows_created": 0,
            "error_message": None,
        }

        try:
            # Create Excel data
            excel_data = self.create_excel_data(structured_data, metadata)
            result["rows_created"] = len(excel_data)

            if not excel_data:
                result["error_message"] = "No data to export"
                return result

            # Convert to DataFrame
            df = pd.DataFrame(excel_data)

            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to Excel
            df.to_excel(output_path, index=False, sheet_name="Output")

            result["success"] = True
            logger.info(f"Excel file generated successfully: {output_path}")

        except Exception as e:
            result["error_message"] = f"Excel generation failed: {str(e)}"
            logger.error(f"Excel generation error: {str(e)}")

        return result

    def generate_comparison_report(
        self, original_data: Dict, extracted_data: Dict, output_path: str
    ) -> Dict[str, Any]:
        """
        Generate a comparison report between expected and extracted data.

        Args:
            original_data (Dict): Expected/reference data
            extracted_data (Dict): LLM extracted data
            output_path (str): Path for comparison report

        Returns:
            Dict[str, Any]: Comparison results
        """
        result = {
            "success": False,
            "output_file": output_path,
            "matches": 0,
            "mismatches": 0,
            "missing_fields": 0,
            "error_message": None,
        }

        try:
            comparison_rows = []
            row_num = 1

            # Header row
            comparison_rows.append(
                {
                    "Field": "Field Name",
                    "Expected": "Expected Value",
                    "Extracted": "Extracted Value",
                    "Match": "Status",
                    "Comments": "Notes",
                }
            )

            # Compare each field
            all_fields = set(list(original_data.keys()) + list(extracted_data.keys()))

            for field in all_fields:
                expected_val = str(original_data.get(field, "")).strip()
                extracted_val = str(extracted_data.get(field, "")).strip()

                # Determine match status
                if field not in original_data:
                    status = "Extra Field"
                    result["missing_fields"] += 1
                elif field not in extracted_data:
                    status = "Missing"
                    result["missing_fields"] += 1
                elif expected_val.lower() == extracted_val.lower():
                    status = "Match"
                    result["matches"] += 1
                else:
                    status = "Mismatch"
                    result["mismatches"] += 1

                comparison_rows.append(
                    {
                        "Field": self.field_mappings.get(
                            field, field.replace("_", " ").title()
                        ),
                        "Expected": expected_val,
                        "Extracted": extracted_val,
                        "Match": status,
                        "Comments": "",
                    }
                )

            # Create DataFrame and save
            df = pd.DataFrame(comparison_rows)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_excel(output_path, index=False, sheet_name="Comparison")

            result["success"] = True
            logger.info(f"Comparison report generated: {output_path}")

        except Exception as e:
            result["error_message"] = f"Comparison report failed: {str(e)}"
            logger.error(f"Comparison error: {str(e)}")

        return result


def generate_excel_from_json(
    json_data: Dict[str, Any], output_path: str, metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate Excel file from JSON data.

    Args:
        json_data (Dict[str, Any]): Structured data to convert
        output_path (str): Output Excel file path
        metadata (Optional[Dict]): Additional metadata

    Returns:
        Dict[str, Any]: Generation results
    """
    generator = ExcelGenerator()
    return generator.generate_excel_file(json_data, output_path, metadata)


def load_and_convert_json(
    json_file_path: str, excel_output_path: str
) -> Dict[str, Any]:
    """
    Load JSON file and convert to Excel format.

    Args:
        json_file_path (str): Path to JSON file with extracted data
        excel_output_path (str): Path for output Excel file

    Returns:
        Dict[str, Any]: Conversion results
    """
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        return generate_excel_from_json(json_data, excel_output_path)

    except Exception as e:
        return {
            "success": False,
            "error_message": f"Failed to load JSON file: {str(e)}",
        }
