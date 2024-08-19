import papermill as pm
from datetime import datetime

file_list = [
    "Getting_Started/ipynb/Pruned by CI + TI/IMDB/Prune by CITI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/IMDB/Prune by CITI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/IMDB/Prune by CITI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/IMDB/Prune by CITI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/OSDG/Prune by CITI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/OSDG/Prune by CITI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/OSDG/Prune by CITI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/OSDG/Prune by CITI(60%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/Yahoo/Prune by CITI(30%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/Yahoo/Prune by CITI(40%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/Yahoo/Prune by CITI(50%).ipynb",
    "Getting_Started/ipynb/Pruned by CI + TI/Yahoo/Prune by CITI(60%).ipynb",
]

for file in file_list:
    # Remove the .ipynb extension and add _saved.ipynb
    base_name = file[:-6]  # Removes the last 6 characters (".ipynb")
    saved_file_name = f"{base_name}_saved.ipynb"

    # Ensure the output directory exists
    pm.execute_notebook(file, saved_file_name)
    print(f"{file} has been saved as {saved_file_name}.")
