import csv

import pandas as pd


def _validate_csv_header(file):
    if not hasattr(file, "seek") or not hasattr(file, "readline"):
        return
    file.seek(0)
    first_line = True

    def decoded_lines():
        nonlocal first_line
        while True:
            line = file.readline()
            if not line:
                return
            if isinstance(line, bytes):
                line = line.decode("utf-8-sig" if first_line else "utf-8")
            first_line = False
            yield line

    try:
        headers = next(csv.reader(decoded_lines()), [])
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("The CSV header is malformed or is not valid UTF-8.") from error
    finally:
        file.seek(0)
    if len(set(headers)) != len(headers):
        raise ValueError("The CSV has duplicate column names; rename them before uploading.")


def load_data(file):
    try:
        _validate_csv_header(file)
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise ValueError(
            "The CSV could not be read. Check that it is a non-empty, well-formed UTF-8 CSV."
        ) from error
    if df.empty or len(df.columns) == 0:
        raise ValueError("The uploaded CSV contains no usable data.")
    return df


def get_basic_info(df):

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(
            df.isnull().sum().sum()
        ),
        "duplicates": int(
            df.duplicated().sum()
        )
    }