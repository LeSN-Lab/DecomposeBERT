import papermill as pm
from datetime import datetime
import os
os.chdir("../../../")

file_list = [
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/Yahoo/Prune by CI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/OSDG/Prune by CI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI/IMDB/Prune by CI(60%).ipynb",
]

script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for file in file_list:
    # Remove the .ipynb extension and add _saved.ipynb
    base_name = file[:-6]  # Removes the last 6 characters (".ipynb")
    saved_file_name = f"{base_name}_saved.ipynb"

    # Ensure the output directory exists
    pm.execute_notebook(file, saved_file_name)
    print(f"{file} has been saved as {saved_file_name}.")
