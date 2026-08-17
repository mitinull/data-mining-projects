from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; REPORT=ROOT/'report'; FIG=REPORT/'figures'

def image_grid(images, path, title, columns=8):
    fig,axes=plt.subplots(int(np.ceil(len(images)/columns)),columns,figsize=(columns,columns*.8));
    for ax,image in zip(axes.flat,images): ax.imshow(image.squeeze(),cmap='gray'); ax.axis('off')
    for ax in axes.flat[len(images):]: ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def main():
    if not (OUT/'results.csv').exists(): raise FileNotFoundError('Run scripts/train.py first.')
    FIG.mkdir(parents=True,exist_ok=True); results=pd.read_csv(OUT/'results.csv'); history=pd.read_csv(OUT/'history.csv'); data=np.load(OUT/'artifacts.npz')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for beta,g in history.groupby('beta'):
        axes[0].plot(g.epoch,g.validation_reconstruction_bce,label=f'β={beta:g}'); axes[1].plot(g.epoch,g.validation_kl,label=f'β={beta:g}'); axes[2].plot(g.epoch,g.validation_elbo,label=f'β={beta:g}')
    for ax,title in zip(axes,['Validation reconstruction BCE','Validation KL','Validation ELBO']): ax.set(title=title,xlabel='Epoch'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'elbo_curves.png',dpi=180); plt.close(fig)
    results.set_index('beta')[['reconstruction_bce','kl','elbo','knn_accuracy','diversity']].plot.bar(subplots=True,layout=(1,5),figsize=(15,3),legend=False); plt.tight_layout(); plt.savefig(FIG/'metric_tradeoff.png',dpi=180); plt.close()
    for beta in results.beta:
        key=f'beta_{beta:g}'; image_grid(data[f'{key}_samples'],FIG/f'samples_{key}.png',f'Random samples, β={beta:g}')
        pairs=np.empty((20,1,28,28)); pairs[0::2]=data[f'{key}_originals']; pairs[1::2]=data[f'{key}_reconstructions'].reshape(-1,1,28,28); image_grid(pairs,FIG/f'recon_{key}.png',f'Original/reconstruction pairs, β={beta:g}',columns=10)
    key='beta_1'; image_grid(data[f'{key}_interpolation'],FIG/'interpolation.png','Latent interpolation, β=1',columns=10); image_grid(data[f'{key}_grid'],FIG/'latent_grid.png','2-D latent grid, β=1',columns=12)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,beta in zip(axes,results.beta):
        key=f'beta_{beta:g}'; m=data[f'{key}_means']; y=data[f'{key}_labels']; s=ax.scatter(m[:,0],m[:,1],c=y,s=2,cmap='tab10'); ax.set(title=f'Latent means, β={beta:g}',xlabel='z₁',ylabel='z₂')
    fig.colorbar(s,ax=axes,label='Digit'); fig.tight_layout(); fig.savefig(FIG/'latent_scatter.png',dpi=180); plt.close(fig)
    lines=[r'\begin{tabular}{rrrrr}',r'\toprule',r'$\beta$ & Recon. BCE & KL & ELBO & k-NN accuracy \\',r'\midrule']
    for _,r in results.iterrows(): lines.append(f"{r.beta:g} & {r.reconstruction_bce:.2f} & {r.kl:.2f} & {r.elbo:.2f} & {r.knn_accuracy:.3f} " + r"\\")
    lines += [r'\bottomrule',r'\end{tabular}']; (REPORT/'generated_results.tex').write_text('\n'.join(lines)+'\n')
    best=results.loc[results.reconstruction_bce.idxmin()]; (REPORT/'generated_summary.tex').write_text(f"The best reconstruction BCE is achieved at $\\beta={best.beta:g}$; the full numerical comparison is generated from the saved test artifacts.\n")
if __name__=='__main__': main()
