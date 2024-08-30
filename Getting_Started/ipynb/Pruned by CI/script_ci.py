import papermill as pm
from datetime import datetime
from pathlib import Path
import nbformat
import os

os.chdir("../../../")

file_list = [
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(60%).ipynb",
]

script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for file in file_list:
    file_path = Path(file)
    print(f"Processing {file_path}...")

    # Record the start time for this notebook
    notebook_start_time = datetime.now()
    print(f"Start time: {notebook_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Read the notebook
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Remove the .ipynb extension and add _saved.ipynb

    # Execute the notebook
    pm.execute_notebook(str(file_path), str(file_path))

    # Record the end time and calculate duration
    notebook_end_time = datetime.now()
    duration = notebook_end_time - notebook_start_time

    # Output the time taken
    print(f"End time: {notebook_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}\n")
