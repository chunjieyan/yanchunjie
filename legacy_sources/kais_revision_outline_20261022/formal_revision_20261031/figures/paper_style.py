"""Shared publication figure style. Encoding meanings are stable across all seven figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
WIDTH_MM=174
FONT=8.5
METHODS={'Control':dict(color='#303030',marker='o',linestyle='-'),'Shuffle':dict(color='#0072B2',marker='s',linestyle='--'),'Ensemble':dict(color='#D55E00',marker='^',linestyle='-.'),'LBSP':dict(color='#00895E',marker='D',linestyle=':')}
BACKBONES={'SASRec':dict(color=METHODS['Control']['color'],marker=METHODS['Control']['marker'],linestyle='-'),'TiSASRec':dict(color='#7B61A8',marker='P',linestyle='--')}
BUCKET_COLORS=('#deebf7','#9ecae1','#4292c6','#08519c')
DATASETS=('ml1m','beauty','sports','toys')
DATASET_LABELS={'ml1m':'MovieLens-1M','beauty':'Beauty','sports':'Sports','toys':'Toys'}
def configure():
    plt.rcParams.update({'font.family':'Arial','font.size':FONT,'axes.labelsize':FONT,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'lines.linewidth':0.9,'pdf.fonttype':42,'ps.fonttype':42,'pdf.compression':9,'axes.unicode_minus':True,'savefig.facecolor':'white','figure.facecolor':'white','mathtext.fontset':'dejavusans'})
def figure(height_mm):
    configure();return plt.figure(figsize=(WIDTH_MM/25.4,height_mm/25.4))
def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,format='pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
