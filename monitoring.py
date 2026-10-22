"""Logs incrementais e renderização de gráficos Braille sem dependências extras."""
from __future__ import annotations
import csv, json, math
from datetime import datetime
from pathlib import Path
from threading import RLock

FIELDS = ['timestamp','run_id','iteration','total_iterations','train_loss','validation_loss','learning_rate','perplexity','best_validation_loss','best_iteration','early_stopping']

class RunLogger:
    def __init__(self, root: Path, start_iteration: int = 0):
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.run_id = f'{stamp}_pid{__import__("os").getpid()}'
        self.directory = root / self.run_id
        self.directory.mkdir()
        self.text_path = self.directory / 'training.log'
        self.csv_path = self.directory / 'metrics.csv'
        self.jsonl_path = self.directory / 'metrics.jsonl'
        self.lock = RLock()
        self.start_iteration = start_iteration
        with self.csv_path.open('w', newline='', encoding='utf-8') as f: csv.DictWriter(f, fieldnames=FIELDS).writeheader()
        self.write(f'Execução {self.run_id} iniciada; iteração inicial acumulada={start_iteration}.')
    def write(self, message: str):
        line = f'[{datetime.now().isoformat(timespec="seconds")}] {message}'
        with self.lock, self.text_path.open('a', encoding='utf-8') as f: f.write(line+'\n'); f.flush()
    def metric(self, **values):
        row = {k:'' for k in FIELDS}; row.update(timestamp=datetime.now().isoformat(timespec='seconds'), run_id=self.run_id); row.update(values)
        with self.lock:
            with self.csv_path.open('a', newline='', encoding='utf-8') as f: csv.DictWriter(f, fieldnames=FIELDS).writerow(row); f.flush()
            with self.jsonl_path.open('a', encoding='utf-8') as f: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
    def close(self, status): self.write(f'Execução encerrada: {status}.')

def load_metrics(root: Path):
    """Carrega o histórico da execução mais recente; ignora linhas incompletas durante escrita concorrente."""
    dirs=sorted((p for p in root.glob('*') if p.is_dir()), reverse=True)
    for d in dirs:
        p=d/'metrics.jsonl'
        if not p.exists(): continue
        rows=[]
        try:
            with p.open(encoding='utf-8') as f:
                for line in f:
                    try: rows.append(json.loads(line))
                    except (json.JSONDecodeError, ValueError): continue
        except OSError: continue
        if rows: return d, rows
    return (dirs[0] if dirs else None), []

def braille_plot(series, width, height, title, x_label='iteração'):
    """series = [(label, [(x,y), ...], marker_char)] -> linhas de texto Braille."""
    width=max(12,min(int(width),160)); height=max(5,min(int(height),50))
    points=[(label,[(float(x),float(y)) for x,y in pts if _finite(x) and _finite(y)], mark) for label,pts,mark in series]
    allp=[p for _,pts,_ in points for p in pts]
    out=[title]
    if not allp: return out+['Sem avaliações registradas ainda.','Use ESC ou Q para voltar.']
    xmin=min(x for x,y in allp); xmax=max(x for x,y in allp); ymin=min(y for x,y in allp); ymax=max(y for x,y in allp)
    if xmin==xmax: xmax=xmin+1
    if ymin==ymax: pad=max(abs(ymin)*.05,.5); ymin-=pad; ymax+=pad
    # Braille cell has 2 columns x 4 rows of dots; raster dimensions in dot units.
    gw=max(8,(width-12)*2); gh=max(8,(height-5)*4); canvas=[[0]*gw for _ in range(gh)]
    masks={'a':1,'b':2,'c':4,'d':8,'e':16,'f':32,'g':64,'h':128,'i':256,'j':512,'k':1024,'l':2048,'m':4096,'n':8192,'o':16384,'p':32768}
    # bit position by dot row/col (Unicode braille dot mapping)
    bitmap=[[0,3],[1,4],[2,5],[6,7]]
    for si,(label,pts,mark) in enumerate(points):
        prev=None
        for x,y in pts:
            px=round((x-xmin)/(xmax-xmin)*(gw-1)); py=round((ymax-y)/(ymax-ymin)*(gh-1)); cur=(px,py)
            if prev:
                steps=max(abs(px-prev[0]),abs(py-prev[1]),1)
                for j in range(steps+1):
                    xx=round(prev[0]+(px-prev[0])*j/steps); yy=round(prev[1]+(py-prev[1])*j/steps); canvas[yy][xx]|=1<<(bitmap[yy%4][xx%2])
            else: canvas[py][px]|=1<<bitmap[py%4][px%2]
            prev=cur
    for cy in range(gh//4):
        line=''
        for cx in range(gw//2):
            bits=0
            for dy in range(4):
                for dx in range(2): bits |= (canvas[cy*4+dy][cx*2+dx] & (1<<bitmap[dy][dx]))
            line+=chr(0x2800+bits)
        out.append(line)
    out += [f'X: {xmin:g} .. {max(x for x,y in allp):g} {x_label} | Y: {min(y for x,y in allp):.5g} .. {max(y for x,y in allp):.5g}', 'Legenda: '+'  '.join(f'{mark} {label}' for label,pts,mark in points)]
    return out

def _finite(v):
    try: return math.isfinite(float(v))
    except (TypeError,ValueError): return False
