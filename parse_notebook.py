import json

with open('data/data recon-wy/RR analysis_basic visual ckeck 1-2-ROI copy.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print(''.join(cell.get('source', [])))
        print('-' * 50)
