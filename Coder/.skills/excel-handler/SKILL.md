# Coder/.skills/excel-handler/SKILL.md
---
name: excel-handler
description: View, inspect, read, edit and duplicate Excel spreadsheets (.xlsx, .xls) using openpyxl, pandas, and xlutils. Activated when tasks require interacting with tabular data or spreadsheet structures. Includes auto-dependency healing.
---
# Excel Handler Skill

This skill allows you to inspect, read, edit, and duplicate sheets in Excel spreadsheets. It delegates operations to executable scripts inside its `scripts` directory.

## Instructions
1. Before making any edits or copies, check the structure of the workbook using `inspect_excel.py`.
2. To read or view data inside a specific sheet, run `read_excel.py`. It outputs data as a Markdown table.
3. To edit values inside a cell, run `edit_excel.py`. This script handles both `.xlsx` and `.xls` files.
4. To duplicate or clone a worksheet within the same workbook, run `duplicate_sheet.py`. This is useful for backups or generating similar report structures.

## Script References
1. **inspect_excel.py**:
   - Format: `inspect_excel.py <file_path>`
   - Example arguments: `["doanh_thu.xlsx"]` or `["data.xls"]`
2. **read_excel.py**:
   - Format: `read_excel.py <file_path> <sheet_name> [start_row] [num_rows]`
   - Example arguments: `["doanh_thu.xlsx", "Sheet1", "0", "15"]`
3. **edit_excel.py**:
   - Format: `edit_excel.py <file_path> <sheet_name> <coordinate> <value>`
   - Example arguments: `["doanh_thu.xlsx", "Sheet1", "C4", "1500"]`
4. **duplicate_sheet.py**:
   - Format: `duplicate_sheet.py <file_path> <sheet_name> [new_sheet_name]`
   - Example arguments: `["DS_ATM.xls", "Tổng hợp", "Tổng hợp (2)"]`