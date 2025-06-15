import json, pandas as pd
from pathlib import Path

# dossier .../project_root/src
SRC_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = SRC_DIR / "assets" / "data"

DATA_PATH    = DATA_DIR / "2024_LoL_esports_match_data_from_OraclesElixir.csv"
IMG_MAP_PATH = DATA_DIR / "champion_images.json"

with IMG_MAP_PATH.open() as f:
    IMG_MAP = json.load(f)

def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    for col in ['position', 'champion', 'result', 'patch']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['win'] = df['result'].map({1: 1, 0: 0})
    df['pick'] = 1
    return df

def _compute_metrics(df, metric, patches, top_n=10):
    metrics = {}
    for p in patches:
        sub = df if p == 'All' else df[df['patch'] == p]
        total = len(sub)
        grp = (
            sub.groupby(['position', 'champion'])
               .agg(wins=('win', 'sum'), picks=('pick', 'sum'))
               .reset_index()
        )
        grp = grp[grp['picks'] >= 5]
        grp['win_rate']  = grp['wins'] / grp['picks']
        grp['pick_rate'] = grp['picks'] / total
        top = (
            grp.sort_values(['position', metric], ascending=[True, False])
               .groupby('position')
               .head(top_n)
        )
        roles  = sorted(top['position'].unique())
        ranks  = [f"Rank {i+1}" for i in range(top_n)]
        z, txt, cd, imgs = [], [], [], []
        for role in roles:
            subr = top[top['position'] == role].nlargest(top_n, metric)
            vals  = subr[metric].tolist()
            names = subr['champion'].tolist()
            games = subr['picks'].tolist()
            while len(vals) < top_n:          # padding
                vals.append(None); names.append(''); games.append(None)
            z.append(vals); txt.append(names); cd.append(games)
            imgs.append([IMG_MAP.get(ch) for ch in names])
        metrics[p] = dict(z=z, txt=txt, cd=cd, imgs=imgs,
                          roles=roles, ranks=ranks)
    return metrics

def precompute():
    df = load_data()
    patches = ['All'] + sorted(df['patch'].unique())
    metrics_map = _compute_metrics(df, 'win_rate',  patches)
    pick_map    = _compute_metrics(df, 'pick_rate', patches)
    return patches, metrics_map, pick_map