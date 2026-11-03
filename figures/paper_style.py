"""Shared publication figure style. Encoding meanings are stable across all seven figures.

Colour roles
  methods / backbones : Okabe-Ito hues, used only for model identity
  tie (same time bucket): one slate ramp, used for every tied-bucket concept
  target item          : one gold highlight
  everything else      : INK, MUTED, REF and two light neutrals
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
WIDTH_MM=174
FONT=8.5
SMALL=7.5
INK='#303030'
MUTED='#6B6B6B'
REF='#8A8A8A'
GRID='#E2E2E2'
RULE='#C8C8C8'
BAND='#F3F3F3'
METHODS={'Control':dict(color=INK,marker='o',linestyle='-'),'Shuffle':dict(color='#0072B2',marker='s',linestyle='--'),'Ensemble':dict(color='#D55E00',marker='^',linestyle='-.'),'LBSP':dict(color='#009E73',marker='D',linestyle=':')}
BACKBONES={'SASRec':dict(color=INK,marker='o',linestyle='-'),'TiSASRec':dict(color='#7B61A8',marker='P',linestyle='--')}
TIE_RAMP=('#E4E9EF','#B5C2D1','#7A8FA8','#44597A')
TIE_FILL,TIE_EDGE,TIE_BG=TIE_RAMP[0],TIE_RAMP[2],'#F4F6F9'
TARGET_FILL,TARGET_EDGE='#FCEFCF','#C4952F'
PREFIX_FILL,PREFIX_EDGE='#F1F1F1','#BDBDBD'
BUCKET_COLORS=TIE_RAMP
DATASETS=('ml1m','beauty','sports','toys')
DATASET_LABELS={'ml1m':'MovieLens-1M','beauty':'Beauty','sports':'Sports','toys':'Toys'}
def configure():
    plt.rcParams.update({'font.family':'Arial','font.size':FONT,'axes.labelsize':FONT,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,
                         'xtick.color':INK,'ytick.color':INK,'lines.linewidth':0.9,'pdf.fonttype':42,'ps.fonttype':42,'pdf.compression':9,'axes.unicode_minus':True,
                         'savefig.facecolor':'white','figure.facecolor':'white',
                         'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.fallback':'stixsans'})
def figure(height_mm):
    configure();return plt.figure(figsize=(WIDTH_MM/25.4,height_mm/25.4))
def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,format='pdf',bbox_inches=None,pad_inches=0);plt.close(fig)
