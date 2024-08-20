import papermill as pm
from datetime import datetime
from pathlib import Path
import nbformat

file_list = [
    "IMDB/Prune by CITI(30%).ipynb",
    "IMDB/Prune by CITI(40%).ipynb",
    "IMDB/Prune by CITI(50%).ipynb",
    "IMDB/Prune by CITI(60%).ipynb",
    "OSDG/Prune by CITI(30%).ipynb",
    "OSDG/Prune by CITI(40%).ipynb",
    "OSDG/Prune by CITI(50%).ipynb",
    "OSDG/Prune by CITI(60%).ipynb",
    "Yahoo/Prune by CITI(30%).ipynb",
    "Yahoo/Prune by CITI(40%).ipynb",
    "Yahoo/Prune by CITI(50%).ipynb",
    "Yahoo/Prune by CITI(60%).ipynb",
]

script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for file in file_list:
    file_path = Path(file)
    print(f"Processing {file_path}...")

    # Record the start time for this notebook
    notebook_start_time = datetime.now()

    # Read the notebook
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Remove the .ipynb extension and add _saved.ipynb
    saved_file_name = file_path.with_name(file_path.stem + "_saved.ipynb")

    # Execute the notebook
    pm.execute_notebook(str(file_path), str(saved_file_name))

    # Record the end time and calculate duration
    notebook_end_time = datetime.now()
    duration = notebook_end_time - notebook_start_time

    # Output the time taken
    print(f"{file_path} has been saved as {saved_file_name}.")
    print(f"Start time: {notebook_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {notebook_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}\n")