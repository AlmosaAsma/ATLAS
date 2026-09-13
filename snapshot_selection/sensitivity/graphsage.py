"""Trained GraphSAGE clustering pipeline for ATLAS LN snapshots."""
from __future__ import annotations
import csv, hashlib, json, os, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.utils import from_networkx, negative_sampling
from tqdm import tqdm

@dataclass(frozen=True)
class Config:
    data_dir: str = './data/snapshots'
    output_dir: str = './results/snapshot_selection/sensitivity/graphsage'
    seed: int = 42
    hidden_dim: int = 64
    embedding_dim: int = 32
    dropout: float = 0.10
    epochs: int = 50
    graphs_per_epoch: int = 20   # 0 means all snapshots each epoch
    max_positive_edges: int = 20000  # 0 means all edges
    negative_ratio: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    min_k: int = 2
    max_k: int = 10
    kmeans_n_init: int = 50
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = Config()
OUT = Path(CFG.output_dir)
LOG = OUT / 'graphsage_pipeline_log.txt'

def log(msg=''):
    print(msg)
    with LOG.open('a', encoding='utf-8') as f: f.write(msg+'\n')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def edge_nodes(e: dict[str, Any]):
    for a,b in [('node1_pub','node2_pub'),('source','target'),('src','dst'),('from','to'),('node1','node2')]:
        if a in e and b in e: return str(e[a]), str(e[b])
    return None

def load_graphs():
    graphs, names, manifest = [], [], []
    p = Path(CFG.data_dir)
    if not p.exists():
        raise FileNotFoundError(f'Missing snapshots directory: {p.resolve()}')

    log(f'Loading graph JSONs from {p.resolve()}...')
    for path in tqdm(sorted(p.glob('*.json')), desc=str(p)):
        row = {'filename':path.name,'path':str(path.resolve()),'sha256':sha256(path),
               'included':False,'reason':'','nodes':0,'edges':0,
               'converted_to_undirected':True,'edge_weights_used':False}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            G = nx.Graph()
            for n in data.get('nodes',[]):
                node_id = n.get('pub_key') or n.get('id') or n.get('node_id')
                if node_id is not None: G.add_node(str(node_id))
            for e in data.get('edges',[]):
                pair = edge_nodes(e)
                if pair: G.add_edge(*pair)
            row['nodes'], row['edges'] = G.number_of_nodes(), G.number_of_edges()
            if G.number_of_nodes() >= 2 and G.number_of_edges() >= 1:
                row['included'] = True; graphs.append(G); names.append(path.name)
            else: row['reason'] = 'too small or edgeless'
        except Exception as exc:
            row['reason'] = f'{type(exc).__name__}: {exc}'
        manifest.append(row)
    log(f'Loaded {len(graphs)} graphs.')
    return graphs, names, manifest

def norm_dict(d):
    m = max(d.values(), default=0.0)
    return {k:(float(v)/m if m>0 else 0.0) for k,v in d.items()}

def add_features(G):
    degree = norm_dict({n:np.log1p(v) for n,v in dict(G.degree()).items()})
    clustering = nx.clustering(G)
    try: core = norm_dict(nx.core_number(G))
    except nx.NetworkXError: core = {n:0.0 for n in G}
    largest = set(max(nx.connected_components(G), key=len)) if len(G) else set()
    for n in G:
        G.nodes[n]['x'] = [degree.get(n,0.0), float(clustering.get(n,0.0)),
                           core.get(n,0.0), 1.0 if n in largest else 0.0]

def to_pyg(graphs):
    result=[]
    for G in tqdm(graphs, desc='Node features'):
        add_features(G)
        d=from_networkx(G); d.x=d.x.float(); result.append(d)
    return result

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=SAGEConv(4, CFG.hidden_dim)
        self.conv2=SAGEConv(CFG.hidden_dim, CFG.embedding_dim)
    def nodes(self,x,edge_index):
        x=F.relu(self.conv1(x,edge_index))
        x=F.dropout(x,p=CFG.dropout,training=self.training)
        return self.conv2(x,edge_index)
    def graph(self,x,edge_index,batch):
        return global_mean_pool(self.nodes(x,edge_index),batch)

def sampled_edges(edge_index, limit):
    if limit<=0 or edge_index.size(1)<=limit: return edge_index
    idx=torch.randperm(edge_index.size(1),device=edge_index.device)[:limit]
    return edge_index[:,idx]

def loss_fn(model,data):
    z=model.nodes(data.x,data.edge_index)
    pos=sampled_edges(data.edge_index,CFG.max_positive_edges)
    nneg=max(1,int(pos.size(1)*CFG.negative_ratio))
    neg=negative_sampling(data.edge_index,num_nodes=data.num_nodes,
                          num_neg_samples=nneg,method='sparse')
    pos_logits=(z[pos[0]]*z[pos[1]]).sum(-1)
    neg_logits=(z[neg[0]]*z[neg[1]]).sum(-1)
    return (F.binary_cross_entropy_with_logits(pos_logits,torch.ones_like(pos_logits))+
            F.binary_cross_entropy_with_logits(neg_logits,torch.zeros_like(neg_logits)))

def train(data_list):
    model=Encoder().to(CFG.device)
    opt=torch.optim.Adam(model.parameters(),lr=CFG.learning_rate,weight_decay=CFG.weight_decay)
    rows=[]
    for epoch in range(1,CFG.epochs+1):
        model.train(); losses=[]
        rng=np.random.default_rng(CFG.seed+epoch)
        idx=rng.permutation(len(data_list)).tolist()
        if CFG.graphs_per_epoch>0: idx=idx[:min(CFG.graphs_per_epoch,len(idx))]
        for i in idx:
            d=data_list[i].to(CFG.device)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model,d)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        mean=float(np.mean(losses)); rows.append((epoch,mean))
        if epoch==1 or epoch%5==0 or epoch==CFG.epochs: log(f'Epoch {epoch:03d}: loss={mean:.6f}')
    with (OUT/'graphsage_training_loss.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['epoch','mean_loss']); w.writerows(rows)
    torch.save({'state_dict':model.state_dict(),'config':asdict(CFG),
                'features':['log_normalised_degree','clustering','normalised_core','largest_cc_indicator']},
               OUT/'graphsage_encoder.pt')
    return model

@torch.no_grad()
def embed(model,data_list):
    model.eval(); vectors=[]
    for d in tqdm(data_list,desc='Graph embeddings'):
        d=d.to(CFG.device)
        batch=torch.zeros(d.num_nodes,dtype=torch.long,device=CFG.device)
        vectors.append(model.graph(d.x,d.edge_index,batch).squeeze(0).cpu().numpy())
    return np.asarray(vectors,dtype=np.float64)

def select_k(X):
    rows=[]; best=None
    for k in range(CFG.min_k,min(CFG.max_k,len(X)-1)+1):
        km=KMeans(n_clusters=k,random_state=CFG.seed,n_init=CFG.kmeans_n_init)
        labels=km.fit_predict(X)
        sil=float(silhouette_score(X,labels)); db=float(davies_bouldin_score(X,labels))
        rows.append((k,sil,db,float(km.inertia_)))
        log(f'k={k}: silhouette={sil:.4f}, DB={db:.4f}, inertia={km.inertia_:.4f}')
        candidate=(sil,-db,k)
        if best is None or candidate>best[0]: best=(candidate,k)
    with (OUT/'graphsage_k_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['k','silhouette','davies_bouldin','inertia']); w.writerows(rows)
    return best[1]

def cluster(names,emb):
    X=StandardScaler().fit_transform(emb)
    k=select_k(X)
    km=KMeans(n_clusters=k,random_state=CFG.seed,n_init=CFG.kmeans_n_init)
    labels=km.fit_predict(X)
    reps={}
    for cid in sorted(np.unique(labels)):
        members=np.where(labels==cid)[0]
        dist=np.linalg.norm(X[members]-km.cluster_centers_[cid],axis=1)
        reps[int(cid)]=int(members[np.argmin(dist)])
    with (OUT/'graphsage_cluster_members.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cluster_id','filename'])
        for name,label in zip(names,labels): w.writerow([int(label),name])
    with (OUT/'graphsage_representatives.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['cluster_id','representative_file','cluster_size','distance_to_centroid'])
        for cid,idx in sorted(reps.items()):
            distance=float(np.linalg.norm(X[idx]-km.cluster_centers_[cid]))
            w.writerow([cid,names[idx],int(np.sum(labels==cid)),distance])
            log(f'Cluster {cid}: {names[idx]} | size={int(np.sum(labels==cid))}')
    return k

def main():
    OUT.mkdir(parents=True,exist_ok=True); LOG.write_text('',encoding='utf-8')
    (OUT/'graphsage_config.json').write_text(json.dumps(asdict(CFG),indent=2),encoding='utf-8')
    set_seed(CFG.seed)
    log('Starting jointly GraphSAGE clustering pipeline.')
    graphs,names,manifest=load_graphs(); log(f'Loaded {len(graphs)} usable graphs.')
    if len(graphs)<3: raise RuntimeError('Need at least three graphs')
    with (OUT/'graphsage_data_manifest.csv').open('w',newline='') as f:
        fields=list(manifest[0]); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)
    data_list=to_pyg(graphs)
    model=train(data_list)
    emb=embed(model,data_list)
    np.save(OUT/'graphsage_embeddings.npy',emb)
    with (OUT/'graphsage_embeddings.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['filename']+[f'embedding_{i}' for i in range(emb.shape[1])])
        for name,v in zip(names,emb): w.writerow([name]+v.tolist())
    log(f'Embedding matrix shape: {emb.shape}')
    k=cluster(names,emb); log(f'Selected k={k}'); log('Finished successfully.')

if __name__=='__main__': main()