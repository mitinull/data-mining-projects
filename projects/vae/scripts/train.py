"""Train a controlled MNIST beta-VAE sweep and save report-ready artifacts."""
import argparse, csv, json, random, gzip, struct, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier

BETAS = [0.5, 1.0, 4.0]
MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)

def load_mnist(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    names=['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for name in names:
        path=data_dir/name
        if not path.exists(): urllib.request.urlretrieve(MNIST_URL+name,path)
    def images(name):
        with gzip.open(data_dir/name,'rb') as f:
            _,count,rows,cols=struct.unpack('>IIII',f.read(16)); return torch.from_numpy(np.frombuffer(f.read(),dtype=np.uint8).copy().reshape(count,1,rows,cols)).float()/255
    def labels(name):
        with gzip.open(data_dir/name,'rb') as f:
            _,count=struct.unpack('>II',f.read(8)); return torch.from_numpy(np.frombuffer(f.read(),dtype=np.uint8).copy())
    return images(names[0]),labels(names[1]),images(names[2]),labels(names[3])

class VAE(nn.Module):
    def __init__(self, latent_size=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 400), nn.ReLU(), nn.Linear(400, 200), nn.ReLU())
        self.mean, self.logvar = nn.Linear(200, latent_size), nn.Linear(200, latent_size)
        self.decoder = nn.Sequential(nn.Linear(latent_size, 200), nn.ReLU(), nn.Linear(200, 400), nn.ReLU(), nn.Linear(400, 784), nn.Sigmoid())
    def encode(self, x):
        h = self.encoder(x.flatten(1)); return self.mean(h), self.logvar(h)
    def forward(self, x):
        mean, logvar = self.encode(x); z = mean + torch.randn_like(mean) * torch.exp(.5 * logvar)
        return self.decoder(z), mean, logvar
    def decode(self, z): return self.decoder(z)

def terms(reconstruction, source, mean, logvar):
    bce = nn.functional.binary_cross_entropy(reconstruction, source.flatten(1), reduction='none').sum(1)
    kl = -.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
    return bce, kl

def evaluate(model, loader, beta, device, collect=False):
    model.eval(); bces=[]; kls=[]; means=[]; labels=[]; originals=[]; reconstructions=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); reconstruction, mean, logvar=model(x); bce,kl=terms(reconstruction,x,mean,logvar)
            bces.extend(bce.cpu().numpy()); kls.extend(kl.cpu().numpy())
            if collect:
                means.append(mean.cpu().numpy()); labels.append(y.numpy()); originals.append(x.cpu().numpy()); reconstructions.append(reconstruction.cpu().numpy())
    result={'reconstruction_bce':float(np.mean(bces)), 'kl':float(np.mean(kls)), 'elbo':float(np.mean(bces)+beta*np.mean(kls))}
    if collect: result.update(means=np.concatenate(means), labels=np.concatenate(labels), originals=np.concatenate(originals), reconstructions=np.concatenate(reconstructions))
    return result

def train_one(beta, train_loader, validation_loader, args, device):
    model=VAE().to(device); opt=torch.optim.Adam(model.parameters(), lr=args.learning_rate); history=[]; best=None; best_loss=float('inf'); stale=0
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,_ in train_loader:
            x=x.to(device); rec,mean,logvar=model(x); bce,kl=terms(rec,x,mean,logvar); loss=(bce+beta*kl).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        validation=evaluate(model,validation_loader,beta,device)
        history.append({'beta':beta,'epoch':epoch,'train_elbo':float(np.mean(losses)),**{'validation_'+k:v for k,v in validation.items()}})
        print(f'beta={beta}: epoch {epoch}, validation ELBO {validation["elbo"]:.2f}')
        if validation['elbo'] < best_loss: best_loss=validation['elbo']; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else:
            stale += 1
            if stale >= args.patience: break
    model.load_state_dict(best); return model.cpu(),history

def make_artifacts(model, test_loader, beta, device):
    test=evaluate(model.to(device),test_loader,beta,device,collect=True); model.cpu()
    means,labels=test['means'],test['labels']; split=len(means)//2
    knn=KNeighborsClassifier(n_neighbors=15).fit(means[:split],labels[:split]); test['knn_accuracy']=float(knn.score(means[split:],labels[split:]))
    with torch.no_grad():
        samples=model.decode(torch.randn(64,2)).reshape(-1,1,28,28).numpy()
        diversity=float(np.mean(np.linalg.norm(samples.reshape(64,-1)[:,None]-samples.reshape(64,-1)[None,:],axis=2)))
        a,b=means[0],means[np.where(labels != labels[0])[0][0]]; path=np.linspace(a,b,10,dtype=np.float32); interpolation=model.decode(torch.from_numpy(path)).reshape(-1,1,28,28).numpy()
        grid=np.stack(np.meshgrid(np.linspace(-3,3,12),np.linspace(-3,3,12)),axis=-1).reshape(-1,2).astype(np.float32); latent_grid=model.decode(torch.from_numpy(grid)).reshape(-1,1,28,28).numpy()
    test.update(samples=samples,interpolation=interpolation,latent_grid=latent_grid,diversity=diversity); return test

def main():
    root=Path(__file__).resolve().parents[1]; p=argparse.ArgumentParser(); p.add_argument('--data-dir',type=Path,default=root/'data'); p.add_argument('--output-dir',type=Path,default=root/'outputs'); p.add_argument('--epochs',type=int,default=15); p.add_argument('--batch-size',type=int,default=256); p.add_argument('--learning-rate',type=float,default=1e-3); p.add_argument('--patience',type=int,default=3); p.add_argument('--seed',type=int,default=42); p.add_argument('--train-size',type=int,default=55000); p.add_argument('--validation-size',type=int,default=5000); p.add_argument('--test-size',type=int,default=10000); args=p.parse_args(); seed_everything(args.seed)
    train_x,train_y,test_x,test_y=load_mnist(args.data_dir); train=TensorDataset(train_x[:args.train_size],train_y[:args.train_size]); val=TensorDataset(train_x[55000:55000+args.validation_size],train_y[55000:55000+args.validation_size]); test=TensorDataset(test_x[:args.test_size],test_y[:args.test_size])
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True); val_loader=DataLoader(val,batch_size=args.batch_size); test_loader=DataLoader(test,batch_size=args.batch_size); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=[]; history=[]; arrays={}; model_dir=args.output_dir/'models'; model_dir.mkdir(parents=True,exist_ok=True)
    for beta in BETAS:
        model,h=train_one(beta,train_loader,val_loader,args,device); artifact=make_artifacts(model,test_loader,beta,device); history.extend(h); rows.append({'beta':beta,**{k:artifact[k] for k in ['reconstruction_bce','kl','elbo','knn_accuracy','diversity']}}); torch.save(model.state_dict(),model_dir/f'beta_{beta:g}.pt')
        key=f'beta_{beta:g}'; arrays.update({f'{key}_samples':artifact['samples'],f'{key}_interpolation':artifact['interpolation'],f'{key}_grid':artifact['latent_grid'],f'{key}_means':artifact['means'],f'{key}_labels':artifact['labels'],f'{key}_originals':artifact['originals'][:10],f'{key}_reconstructions':artifact['reconstructions'][:10]})
    with (args.output_dir/'results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with (args.output_dir/'history.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
    np.savez_compressed(args.output_dir/'artifacts.npz',**arrays)
    (args.output_dir/'config.json').write_text(json.dumps(vars(args)|{'betas':BETAS},indent=2,default=str)); print('Saved real beta-VAE artifacts.')
if __name__=='__main__': main()
