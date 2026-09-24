# Unified, reproducible SAR-VAE vs SAR-AAE experiment
# Uses the existing Wakashio split CSV and deterministic VAE evaluation.
# DARTIS split follows the project methodology: 200/category, fixed random split,
# normal-only training, oil only in validation/test.

import os, random, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, jaccard_score

SEEDS = [42, 123, 456, 789, 999]
IMG_SIZE = 256
BATCH_SIZE = 16
LATENT_DIM = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NpyDataset(Dataset):
    def __init__(self, root, rows):
        self.root = Path(root); self.rows = rows.reset_index(drop=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = np.load(self.root / r.filename).astype(np.float32)
        x = torch.from_numpy(x)[None]
        x = nn.functional.interpolate(x[None], size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[0]
        # Keep preprocessing consistent with [0,1] sigmoid decoder assumption.
        if x.max() > 1.0 or x.min() < 0.0:
            lo, hi = x.min(), x.max()
            if hi > lo: x = (x - lo) / (hi - lo)
        return x, int(r.label)


class JpgDataset(Dataset):
    def __init__(self, root, rows):
        self.root = Path(root); self.rows = rows.reset_index(drop=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = np.asarray(Image.open(self.root / r.filename).convert('L'), dtype=np.float32) / 255.0
        x = torch.from_numpy(x)[None]
        x = nn.functional.interpolate(x[None], size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[0]
        return x, int(r.label)


class SAR_VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(), nn.Flatten())
        self.fc_mu = nn.Linear(64*64*64, latent_dim)
        self.fc_var = nn.Linear(64*64*64, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64*64*64)
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(64,64,64)),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid())

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        std = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(std)*std
        return self.decoder(self.decoder_input(z)), mu, logvar

    def reconstruct(self, x, deterministic=True):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5*logvar)
            z = mu + torch.randn_like(std)*std
        return self.decoder(self.decoder_input(z))

    def compute_loss(self, x):
        recon, mu, logvar = self.forward(x)
        mse = nn.functional.mse_loss(recon, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld


class SAR_AAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64*64*64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,64*64*64), nn.Unflatten(1,(64,64,64)),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,4,2,1), nn.Sigmoid())
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
    def forward_reconstruction(self,x):
        z=self.encoder(x); return self.decoder(z),z
    def reconstruct(self,x): return self.forward_reconstruction(x)[0]


def train_vae(loader, seed, max_epochs=100, patience=10):
    set_seed(seed); model=SAR_VAE().to(DEVICE); opt=optim.Adam(model.parameters(),lr=1e-3)
    best=float('inf'); bad=0
    for ep in range(max_epochs):
        model.train(); total=0
        for imgs,_ in loader:
            imgs=imgs.to(DEVICE); opt.zero_grad(); loss=model.compute_loss(imgs)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            total += loss.item()
        avg=total/len(loader.dataset)
        if avg < best-1e-4: best,bad=avg,0
        else: bad+=1
        if bad>=patience: break
    return model


def train_aae(loader, seed, epochs=50):
    set_seed(seed); model=SAR_AAE().to(DEVICE)
    opt_r=optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters()),lr=1e-3)
    opt_d=optim.Adam(model.discriminator.parameters(),lr=1e-4)
    opt_g=optim.Adam(model.encoder.parameters(),lr=1e-4)
    bce,mse=nn.BCELoss(),nn.MSELoss(); model.train()
    for _ in range(epochs):
        for imgs,_ in loader:
            imgs=imgs.to(DEVICE); valid=torch.ones((len(imgs),1),device=DEVICE); fake=torch.zeros((len(imgs),1),device=DEVICE)
            opt_r.zero_grad(); recon,_=model.forward_reconstruction(imgs); lr=mse(recon,imgs); lr.backward(); opt_r.step()
            opt_d.zero_grad(); zr=torch.randn((len(imgs),LATENT_DIM),device=DEVICE); zf=model.encoder(imgs).detach()
            ld=bce(model.discriminator(zr),valid)+bce(model.discriminator(zf),fake); ld.backward(); opt_d.step()
            opt_g.zero_grad(); zf2=model.encoder(imgs); lg=bce(model.discriminator(zf2),valid); lg.backward(); opt_g.step()
    return model


def errors(model, loader, vae=False):
    model.eval(); e=[]; y=[]
    with torch.no_grad():
        for x,lbl in loader:
            x=x.to(DEVICE); r=model.reconstruct(x, deterministic=True) if vae else model.reconstruct(x)
            er=nn.functional.mse_loss(r,x,reduction='none').flatten(1).mean(1)
            e.extend(er.cpu().numpy()); y.extend(lbl.numpy())
    return np.asarray(e),np.asarray(y)


def threshold_and_polarity(err,y):
    raw_auc=roc_auc_score(y,err)
    # high-error anomaly if validation AUC >= 0.5; otherwise invert polarity.
    polarity='high_error' if raw_auc>=0.5 else 'low_error'
    score=err if polarity=='high_error' else -err
    fpr,tpr,thr=roc_curve(y,score)
    k=np.argmax(tpr-fpr)
    return float(thr[k]),polarity,float(raw_auc)


def evaluate(model, loader, threshold, polarity, vae=False):
    e,y=errors(model,loader,vae)
    score=e if polarity=='high_error' else -e
    pred=(score>threshold).astype(int)
    return {'auc':float(roc_auc_score(y,score)), 'iou':float(jaccard_score(y,pred,zero_division=0)), 'mse':float(e.mean())}


def run_dataset(name, train_loader, val_loader, test_loader, outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for seed in SEEDS:
        print(f'[{name}] seed {seed}')
        vae=train_vae(train_loader,seed); aae=train_aae(train_loader,seed)
        ev,yv=errors(vae,val_loader,vae=True); ea,_=errors(aae,val_loader)
        tv,pv,rawv=threshold_and_polarity(ev,yv); ta,pa,rawa=threshold_and_polarity(ea,yv)
        rv=evaluate(vae,test_loader,tv,pv,vae=True); ra=evaluate(aae,test_loader,ta,pa)
        for model,res,pol,raw,thr in [('VAE',rv,pv,rawv,tv),('AAE',ra,pa,rawa,ta)]:
            row=dict(dataset=name,seed=seed,model=model,validation_raw_auc=raw,polarity=pol,threshold=thr,**res)
            allrows.append(row); print(row)
        torch.save(vae.state_dict(),outdir/f'{name}_vae_seed{seed}.pth')
        torch.save(aae.state_dict(),outdir/f'{name}_aae_seed{seed}.pth')
    with open(outdir/f'{name}_results.json','w') as f: json.dump(allrows,f,indent=2)
    return allrows


def load_wakashio(base):
    base=Path(base); csv=__import__('pandas').read_csv(base/'wakashio_labels_split.csv')
    patchdir=base/'dataset_wakashio/patches'
    # Only files explicitly present in the split CSV are experimental inputs.
    loaders=[]
    for split in ['train','val','test']:
        ds=NpyDataset(patchdir,csv[csv.split==split])
        loaders.append(DataLoader(ds,batch_size=BATCH_SIZE,shuffle=(split=='train'),num_workers=0))
    return loaders


def load_dartis(base):
    import pandas as pd
    root=Path(base)/'dataset_DARTIS_2019/subset_images'
    files=sorted([p.name for p in root.glob('*.jpg')])
    cats={'ow':[],'nw':[],'oc':[],'nc':[]}
    for f in files:
        cats[f.split('-')[0]].append(f)
    # Reproduce the project sampling logic: 200/category with fixed seed.
    rng=random.Random(42); chosen={k:rng.sample(v,200) for k,v in cats.items()}
    normal=chosen['nw']+chosen['nc']; oil=chosen['ow']+chosen['oc']
    rng.shuffle(normal); rng.shuffle(oil)
    ntr=int(.60*len(normal)); nval=int(.20*len(normal))
    train=normal[:ntr]
    val=normal[ntr:ntr+nval]+oil[:140]
    test=normal[ntr+nval:]+oil[140:]
    def df(xs,label_fn): return pd.DataFrame({'filename':xs,'label':[label_fn(x) for x in xs]})
    tr=df(train,lambda x:0); va=df(val,lambda x:1 if x[:2] in ('ow','oc') else 0); te=df(test,lambda x:1 if x[:2] in ('ow','oc') else 0)
    return [DataLoader(JpgDataset(root,d),batch_size=BATCH_SIZE,shuffle=(i==0),num_workers=0) for i,d in enumerate([tr,va,te])]


if __name__=='__main__':
    BASE=os.environ.get('SAR_BASE','/content/sar_backup_inspect')
    OUT=os.environ.get('SAR_OUT','/content/sar_results_final')
    w=load_wakashio(BASE); run_dataset('Wakashio',*w,OUT)
    d=load_dartis(BASE); run_dataset('DARTIS_2019',*d,OUT)
    print('DONE')
