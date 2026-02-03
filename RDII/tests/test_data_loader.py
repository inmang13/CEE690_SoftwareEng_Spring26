""" 
Unit tests for data_loader.py
"""

import pytest
import pandas as pd
import os
from rdii.data_loader import read_flow_meter_data, read_all_flow_meters, extract_meter_name

# TEST extract_meter_name

def test_extract_meter_name():
    filename='DURHAM_DBO_20230101-20260101.csv'
    result=extract_meter_name(filename)
    assert result=='DBO'

def test_extract_meter_name_invalid_pattern():
    """Test error handling for invalid filename pattern."""
    filename = "invalid_file.csv"
    with pytest.raises(ValueError, match="doesn't match expected pattern"):
        extract_meter_name(filename)

if __name__ == "__main__":
    pytest.main()