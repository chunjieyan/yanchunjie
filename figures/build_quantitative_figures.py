"""Rebuild Figs4-7 from seven unrounded derived CSV subsets; no model/data access.

Usage: python build_quantitative_figures.py --data-root inputs --output-dir .
Requires matplotlib and the sibling paper_style.py. PDFs retain vector text.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FuncFormatter,FixedLocator
import paper_style as ps

HERE=Path(__file__).resolve().parent
METHODS=('Control','Shuffle','Ensemble','LBSP')
SEEDS=(42,2026,3407)
SEED_LINES={42:'-',2026:(0,(4,2)),3407:(0,(1,1.6))}
INPUTS={};AUDIT=[];GEOMETRY=[]

def read(root,name):
 candidates=[root/name,root/'statistics'/name,root/'X7/training'/name]
 p=next((p for p in candidates if p.is_file()),None)
 if p is None:raise FileNotFoundError(name+' missing from --data-root')
 INPUTS[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def one(rows,**where):
 chosen=[r for r in rows if all(r[k]==str(v) for k,v in where.items())]
 if len(chosen)!=1:raise ValueError('Expected one row '+str(where)+', found '+str(len(chosen)))
 return chosen[0]
def grid(ax):ax.grid(axis='y',color='#dedede',lw=.45,zorder=0);ax.set_axisbelow(True);ax.tick_params(length=2.5,width=.6,pad=2.5)
def save(fig,out,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();width,height=fig.get_size_inches()*fig.dpi
 # Exported labels are audited for full-canvas bounds, independently of subplot limits.
 bounds=[]
 for text in fig.findobj(match=lambda obj:hasattr(obj,'get_text')):
  if not getattr(text,'get_visible',lambda:False)() or not getattr(text,'get_text',lambda:'')():continue
  box=text.get_window_extent(renderer)
  if box.width==0 or box.height==0:continue
  bounds.append({'text':text.get_text(),'bbox_px':[box.x0,box.y0,box.x1,box.y1]})
  if min(box.x0,box.y0)<-.8 or box.x1>width+.8 or box.y1>height+.8:raise ValueError('Canvas text clipping '+name+': '+text.get_text())
 fig.savefig(out/(name+'.png'),dpi=220,bbox_inches=None,pad_inches=0)
 GEOMETRY.append({'figure':name,'width_mm':ps.WIDTH_MM,'height_mm':float(fig.get_size_inches()[1]*25.4),'text_bounds_checked':len(bounds),'canvas_clipping':False})
 ps.save(fig,out/(name+'.pdf'))

def figure4(root,out):
 data=read(root,'bucket_stability_all_users_summary.csv');refs=read(root,'seed_reference_summary.csv')
 fig=ps.figure(63);axes=fig.subplots(1,4,sharey=True);fig.subplots_adjust(left=.079,right=.991,bottom=.285,top=.795,wspace=.21)
 handles=[Line2D([0],[0],label=name,markersize=4,color=s['color'],marker=s['marker'],linestyle=s['linestyle']) for name,s in ps.BACKBONES.items()]
 fig.legend(handles=handles,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.54,1.012),handlelength=2.1,columnspacing=2.5)
 for i,(ax,ds) in enumerate(zip(axes,ps.DATASETS)):
  ax.set_title('('+chr(97+i)+') '+ps.DATASET_LABELS[ds],loc='left',pad=4,fontweight='bold');counts=None
  for name,method,offset in [('SASRec','Control',-.055),('TiSASRec','TiSASRec',.055)]:
   rows=[r for r in data if r['dataset']==ds and r['method']==method]
   if not rows:
    if method!='TiSASRec' or ds not in ['sports','toys']:raise ValueError('Missing Fig4 series')
    continue
   s=ps.BACKBONES[name];rr=[one(rows,bucket=b,metric='overlap10') for b in ['2','3','4','>=5']];n=[int(r['n']) for r in rr]
   if counts is None:counts=n
   elif counts!=n:raise ValueError('Backbone bucket populations differ')
   x=np.arange(1,5)+offset;y=np.array([float(r['mean'])*100 for r in rr]);sd=np.array([float(r['sample_sd'])*100 for r in rr])
   if min(y-sd)<0 or max(y+sd)>100:raise ValueError('Fig4 meanSD outside plotted full percentage range')
   ax.errorbar(x,y,yerr=sd,color=s['color'],marker=s['marker'],ms=3.6,lw=.9,linestyle=s['linestyle'],capsize=1.8,elinewidth=.7,zorder=3)
   ref=one(refs,dataset=ds,method=method,population='full');value=float(ref['mean_overlap10'])*100
   ax.hlines(value,.58,4.38,color=s['color'],linestyle=(0,(1.5,2.0)),lw=.8,zorder=1)
   # Short right-margin leaders keep labels off the horizontal reference and observed data.
   if ds=='ml1m':label_y,va=(67,'bottom') if method=='Control' else (49,'top')
   elif ds=='beauty':label_y,va=(11,'top') if method=='Control' else (52,'bottom')
   else:label_y,va=12,'top'
   ax.annotate('seed '+f'{value:.1f}'+'%',xy=(4.38,value),xytext=(4.20,label_y),ha='right',va=va,fontsize=8,color=s['color'],arrowprops=dict(arrowstyle='-',lw=.65,color=s['color'],connectionstyle='angle,angleA=0,angleB=90,rad=0'),annotation_clip=False)
   for r,xx,yy,ss in zip(rr,x,y,sd):AUDIT.append({'figure':4,'dataset':ds,'method':method,'bucket':r['bucket'],'n':int(r['n']),'mean':float(r['mean']),'sample_sd':float(r['sample_sd']),'plotted_x':float(xx),'plotted_y_percent':float(yy),'seed_reference_full_panel':float(ref['mean_overlap10'])})
  ax.set_xlim(.55,4.55);ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([1,2,3,4],labels=[b+'\n'+str(n) for b,n in zip(['2','3','4','≥5'],counts)]);grid(ax)
 fig.text(.013,.55,'Top-10 overlap (%)',ha='center',va='center',rotation=90,fontsize=8.5)
 fig.text(.54,.025,'Last-bucket size (stratum N on second line)',ha='center',fontsize=8.5)
 save(fig,out,'Fig4')

def nice_limits(lo,hi):
 span=max(hi-lo,1.0);raw=span/4;power=10**math.floor(math.log10(raw));step=next(v*power for v in [1,2,5,10] if v*power>=raw);step=max(1,step)
 low=math.floor((min(lo,0)-.08*span)/step)*step;high=math.ceil((max(hi,0)+.08*span)/step)*step
 return low,high,step

def figure5(root,out):
 means=read(root,'method_summary.csv');paired=read(root,'paired_difference_summary.csv');fig=ps.figure(94);axes=fig.subplots(2,2);fig.subplots_adjust(left=.12,right=.985,bottom=.13,top=.84,hspace=.50,wspace=.30)
 handles=[Line2D([0],[0],label=m,marker=ps.METHODS[m]['marker'],color=ps.METHODS[m]['color'],linestyle='None',markersize=4.5) for m in METHODS]
 fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.54,1.004),handletextpad=.45,columnspacing=1.75)
 for i,(ax,ds) in enumerate(zip(axes.flat,ps.DATASETS)):
  rr=[(m,one(means,dataset=ds,method=m,population='affected',metric='overlap10'),one(paired,dataset=ds,method=m,population='affected',metric='ndcg10')) for m in METHODS]
  lo=min(float(y['ci95_lower'])*1000 for _,_,y in rr);hi=max(float(y['ci95_upper'])*1000 for _,_,y in rr);yl,yu,step=nice_limits(lo,hi)
  ax.set_xlim(0,107);ax.set_ylim(yl,yu);ax.set_xticks([0,25,50,75,100]);ax.set_yticks(np.arange(yl,yu+step*.1,step));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0f}'));grid(ax);ax.axhline(0,color='#777777',lw=.7,ls=(0,(3,2)),zorder=1)
  ax.set_title('('+chr(97+i)+') '+ps.DATASET_LABELS[ds]+' (N='+f'{int(rr[0][1]["n"]):,}'+')',loc='left',fontweight='bold',pad=4)
  for m,x,y in rr:
   s=ps.METHODS[m];xm=float(x['mean'])*100;ym=float(y['mean_delta'])*1000;xlo=float(x['ci95_lower'])*100;xhi=float(x['ci95_upper'])*100;ylo=float(y['ci95_lower'])*1000;yhi=float(y['ci95_upper'])*1000
   if not(0<=xlo<=xhi<=107 and yl<=ylo<=yhi<=yu):raise ValueError('Fig5CI would be clipped')
   ax.errorbar(xm,ym,xerr=[[xm-xlo],[xhi-xm]],yerr=[[ym-ylo],[yhi-ym]],fmt='none',color=s['color'],elinewidth=.8,capsize=2,zorder=2)
   ax.plot([xm],[ym],marker=s['marker'],color=s['color'],ms=4.5,linestyle='None',zorder=3)
   AUDIT.append({'figure':5,'dataset':ds,'method':m,'n':int(x['n']),'overlap':float(x['mean']),'delta':float(y['mean_delta']),'x_ci':[float(x['ci95_lower']),float(x['ci95_upper'])],'y_ci':[float(y['ci95_lower']),float(y['ci95_upper'])],'plotted_y_multiplier':1000,'axis_limits_x_percent':[0,107],'axis_limits_y_milli':[yl,yu],'fixed_marker_size_pt':4.5,'cost_encoding':False})
 fig.text(.018,.49,r'ΔNDCG@10 ($\times10^{-3}$)',rotation=90,ha='center',va='center',fontsize=8.5);fig.text(.55,.018,'Affected-user Top-10 overlap (%)',ha='center',fontsize=8.5)
 save(fig,out,'Fig5')

def figure6(root,out):
 data=read(root,'paired_per_seed_user_bootstrap.csv');fig=ps.figure(100);axes=fig.subplots(2,2,sharey=True);fig.subplots_adjust(left=.17,right=.985,bottom=.12,top=.91,hspace=.44,wspace=.21)
 methods=('Shuffle','Ensemble','LBSP');ys={(m,s):9-j*3.45-k for j,m in enumerate(methods) for k,s in enumerate(SEEDS)}
 for i,(ax,ds) in enumerate(zip(axes.flat,ps.DATASETS)):
  rr=[one(data,dataset=ds,method=m,training_seed=s,population='affected',metric='ndcg10') for m in methods for s in SEEDS];lo=min(float(r['ci95_lower'])*1000 for r in rr);hi=max(float(r['ci95_upper'])*1000 for r in rr);xl,xu,step=nice_limits(lo,hi)
  ax.set_xlim(xl,xu);ax.set_ylim(-.65,9.8);ax.set_xticks(np.arange(xl,xu+step*.1,step));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.0f}'));ax.tick_params(length=2.5,width=.6,pad=2.5);ax.axvline(0,color='#777777',ls=(0,(3,2)),lw=.7,zorder=1)
  ax.set_title('('+chr(97+i)+') '+ps.DATASET_LABELS[ds],loc='left',fontweight='bold',pad=4)
  for m in methods:
   color=ps.METHODS[m]['color'];marker=ps.METHODS[m]['marker']
   for seed in SEEDS:
    r=one(data,dataset=ds,method=m,training_seed=seed,population='affected',metric='ndcg10');y=ys[m,seed];mean=float(r['mean_delta'])*1000;low=float(r['ci95_lower'])*1000;high=float(r['ci95_upper'])*1000
    ax.hlines(y,low,high,color=color,lw=.9,linestyle=SEED_LINES[seed],zorder=2);ax.vlines([low,high],y-.14,y+.14,color=color,lw=.75,zorder=2);ax.plot(mean,y,color=color,marker=marker,ms=3.7,linestyle='None',zorder=3)
    AUDIT.append({'figure':6,'dataset':ds,'method':m,'training_seed':seed,'n':int(r['n']),'delta':float(r['mean_delta']),'ci':[float(r['ci95_lower']),float(r['ci95_upper'])],'bootstrap_replicates':int(r['bootstrap_replicates']),'plotted_multiplier':1000,'axis_limits_milli':[xl,xu]})
   if i%2==0:ax.text(-.21,ys[m,2026],m,transform=ax.get_yaxis_transform(),ha='right',va='center',color=color,fontsize=8.5)
  ax.set_yticks([ys[m,s] for m in methods for s in SEEDS],labels=[str(s) for m in methods for s in SEEDS]);ax.tick_params(axis='y',length=0,labelleft=(i%2==0));ax.spines['left'].set_visible(False)
 fig.text(.56,.02,r'Affected-user ΔNDCG@10 ($\times10^{-3}$; 95% paired-user bootstrap CI)',ha='center',fontsize=8.5)
 save(fig,out,'Fig6')

def figure7(root,out):
 data=read(root,'epoch_validation_curves.csv');summaries=read(root,'training_run_summary.csv');fig=ps.figure(94);axes=fig.subplots(2,2,sharex=True,sharey='row');fig.subplots_adjust(left=.12,right=.985,bottom=.115,top=.83,hspace=.40,wspace=.20)
 handles=[Line2D([0],[0],label='Seed '+str(s),color='#303030',linestyle=SEED_LINES[s],lw=1) for s in SEEDS]+[Line2D([0],[0],label='Best validation',color='#303030',marker='*',markersize=7,linestyle='None'),Line2D([0],[0],label='Stopped',color='#303030',marker='|',markersize=7,linestyle='None')]
 fig.legend(handles=handles,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(.55,1.006),columnspacing=1,handletextpad=.4,handlelength=1.7)
 loss=[float(r['train_loss']) for r in data if r['dataset']=='sports'];span=max(loss)-min(loss);loss_lim=(min(loss)-span*.06,max(loss)+span*.06);min_epoch=min(int(r['epoch']) for r in data);max_epoch=max(int(r['epoch']) for r in data)
 for col,method in enumerate(('Control','LBSP')):
  color=ps.METHODS[method]['color']
  for row,field in enumerate(('val_ndcg10','train_loss')):
   ax=axes[row,col];ax.set_title('('+chr(97+row*2+col)+') '+method,loc='left',fontweight='bold',pad=4,color=color);grid(ax);ax.set_xlim(min_epoch,max_epoch)
   for seed in SEEDS:
    rr=sorted([r for r in data if r['dataset']=='sports' and r['method']==method and int(r['training_seed'])==seed],key=lambda r:int(r['epoch']));summary=one(summaries,dataset='sports',method=method,training_seed=seed);best=int(summary['best_epoch']);stop=int(summary['epochs_run']);selected=one(rr,epoch=best)
    if [int(r['epoch']) for r in rr]!=list(range(1,stop+1)):raise ValueError('Missing recorded epoch')
    ax.plot([int(r['epoch']) for r in rr],[float(r[field]) for r in rr],color=color,ls=SEED_LINES[seed],lw=.9,zorder=2)
    ax.plot(best,float(selected[field]),marker='*',color=color,markersize=6,linestyle='None',zorder=4)
    ax.plot(stop,float(rr[-1][field]),marker='|',color=color,markersize=7,markeredgewidth=1.2,linestyle='None',zorder=4,clip_on=False)
    if row==0:AUDIT.append({'figure':7,'method':method,'training_seed':seed,'best_epoch':best,'stop_epoch':stop,'epochs':len(rr),'best_val_ndcg10':float(selected['val_ndcg10']),'last_val_ndcg10':float(rr[-1]['val_ndcg10']),'last_train_loss':float(rr[-1]['train_loss']),'all_recorded_epochs_plotted':True})
   if row==0:ax.set_ylim(0,.015);ax.set_yticks([0,.005,.010,.015]);ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.3f}'))
   else:ax.set_ylim(*loss_lim);ax.yaxis.set_major_locator(MaxNLocator(nbins=4));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:.1f}'));ax.set_xlabel('Epoch',labelpad=2.5)
   ax.set_xticks([min_epoch,20,40,60,max_epoch])
 axes[0,0].set_ylabel('Validation NDCG@10',labelpad=5);axes[1,0].set_ylabel('Training loss',labelpad=5)
 save(fig,out,'Fig7')
 GEOMETRY[-1].update({'recorded_epoch_range':[min_epoch,max_epoch],'loss_axis_limits':list(loss_lim),'zero_origin_for_loss':False,'seed_encoding':'line style only','checkpoint_marker':'star','stopping_marker':'vertical tick'})

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',type=Path,default=HERE/'inputs');parser.add_argument('--output-dir',type=Path,default=HERE);args=parser.parse_args();root=args.data_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 figure4(root,out);figure5(root,out);figure6(root,out);figure7(root,out)
 result={'status':'BUILT_PENDING_PNG_VISUAL_REVIEW','inputs_sha256':INPUTS,'shared_style_sha256':hashlib.sha256((HERE/'paper_style.py').read_bytes()).hexdigest(),'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('Fig[4-7].*')) if p.suffix in ['.pdf','.png']},'geometry':GEOMETRY,'numeric_points':AUDIT,'data_changed_or_rounded':False,'cost_encoding_in_Fig5':False,'raw_interaction_checkpoint_or_GPU_access':False,'portable_cli':'python build_quantitative_figures.py --data-root inputs --output-dir .'}
 (out/'quantitative_figures_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'status':'BUILT','PDFs':4,'points':len(AUDIT)}))
if __name__=='__main__':main()
