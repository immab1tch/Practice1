from pathlib import Path
import json
from model import Params
from figures_en import create_figures
root=Path(__file__).resolve().parent
create_figures(root/'results',root,Params(**json.loads((root/'results/parameters_used.json').read_text())))
print('English M1 figures written to results/figures')
