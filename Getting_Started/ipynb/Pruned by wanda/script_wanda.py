import os

import papermill as pm
from datetime import datetime

file_list = [
    "Getting_Started/ipynb/Pruned by wanda/IMDB/Pruned by wanda(30%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/IMDB/Pruned by wanda(40%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/IMDB/Pruned by wanda(50%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/IMDB/Pruned by wanda(60%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/OSDG/Pruned by wanda(30%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/OSDG/Pruned by wanda(40%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/OSDG/Pruned by wanda(50%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/OSDG/Pruned by wanda(60%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/Yahoo/Pruned by wanda(30%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/Yahoo/Pruned by wanda(40%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/Yahoo/Pruned by wanda(50%).ipynb",
    "Getting_Started/ipynb/Pruned by wanda/Yahoo/Pruned by wanda(60%).ipynb",
]

# Record the script start time
script_start_time = datetime.now()
print(f"Script started at: {script_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for file in file_list:
    # Remove the .ipynb extension and add _saved.ipynb
    base_name = file[:-6]  # Removes the last 6 characters (".ipynb")
    saved_file_name = f"{base_name}_saved.ipynb"

    # Ensure the output directory exists
    pm.execute_notebook(file, saved_file_name)
    print(f"{file} has been saved as {saved_file_name}.")