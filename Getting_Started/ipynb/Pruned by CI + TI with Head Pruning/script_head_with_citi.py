import papermill as pm
from datetime import datetime

file_list = [
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/IMDB/Prune by CI + TI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/IMDB/Prune by CI + TI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/IMDB/Prune by CI + TI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/IMDB/Prune by CI + TI with Head Pruning(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/OSDG/Prune by CI + TI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/OSDG/Prune by CI + TI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/OSDG/Prune by CI + TI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/OSDG/Prune by CI + TI with Head Pruning(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/Yahoo/Prune by CI + TI with Head Pruning(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/Yahoo/Prune by CI + TI with Head Pruning(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/Yahoo/Prune by CI + TI with Head Pruning(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI with Head Pruning/Yahoo/Prune by CI + TI with Head Pruning(60%).ipynb",
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
