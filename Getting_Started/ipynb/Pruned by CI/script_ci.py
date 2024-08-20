import papermill as pm
from datetime import datetime
from pathlib import Path


os.chdir(Path("../../../").resolve())
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
    file_path = Path(file)
    
    # Remove the .ipynb extension and add _saved.ipynb
    saved_file_name = file_path.with_name(file_path.stem + "_saved.ipynb")

    # Ensure the output directory exists
    saved_file_path = file_path.parent / saved_file_name
    
    pm.execute_notebook(str(file_path), str(saved_file_path))
    print(f"{file_path} has been saved as {saved_file_path}.")