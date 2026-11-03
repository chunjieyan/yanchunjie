#!/usr/bin/env python3
"""Portable vector redraw of Figures1–3 from bundled saved-report summaries.

Requires Python3, matplotlib, pypdf and Arial (installed or --font-dir).
The bundled product/counter files contain the exact verified source excerpt;
absolute provenance paths inside JSON are documentation, never dependencies.
No raw interaction data, checkpoint, inference or training code is accessed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
import matplotlib.pyplot as plt
from pypdf import PdfReader
import paper_style as style

HERE=Path(__file__).resolve().parent
INPUTS=HERE/'inputs'
INPUT_SHA={'fig1_example.json':'9a00b484bbf59e6a6efbbb9c65b203b5515dc0c152f58af6f2510a3c980e0341',
           'fig3_counts.csv':'a63d04dc6bbf8b8b13a94c4af79851bc1f2b155b35eda90c511ae6278135dbda'}
HEIGHTS={'Fig1':66,'Fig2':86,'Fig3':62}
GRAY=style.INK
MUTED=style.MUTED
BUCKET_FILL,BUCKET_EDGE=style.TIE_FILL,style.TIE_EDGE
PREFIX_FILL,PREFIX_EDGE=style.PREFIX_FILL,style.PREFIX_EDGE
TARGET_FILL,TARGET_EDGE=style.TARGET_FILL,style.TARGET_EDGE
EXCLUDED_EDGE=style.MUTED


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font_setup(directory=None):
    candidates=[Path(directory)] if directory else [Path('C:/Windows/Fonts')]
    for folder in candidates:
        for name in ('arial.ttf','arialbd.ttf','Arial.ttf','Arial Bold.ttf'):
            p=folder/name
            if p.exists():font_manager.fontManager.addfont(str(p))
    resolved=Path(font_manager.findfont(font_manager.FontProperties(family='Arial'),fallback_to_default=False))
    return {'family':'Arial','body_pt':style.FONT,'resolved_regular_font_name':resolved.name,'font_sha256':sha(resolved)}


def inputs():
    for name,expected in INPUT_SHA.items():assert sha(INPUTS/name)==expected,name
    case=json.loads((INPUTS/'fig1_example.json').read_text(encoding='utf-8'))
    with (INPUTS/'fig3_counts.csv').open(encoding='utf-8-sig',newline='') as f:counts=list(csv.DictReader(f))
    assert [r['dataset'] for r in counts]==list(style.DATASETS)
    orders=case['orders'];assert [o['target_rank'] for o in orders]==[1,335]
    assert len(set(orders[0]['top10_asins'])&set(orders[1]['top10_asins']))==case['full_top10_intersection_count']==0
    assert orders[0]['last_bucket_asins']==list(reversed(orders[1]['last_bucket_asins'])) and case['fixed_prefix_events']==21
    assert all(len(o['top10_asins'])==10 and o['display_top3_asins']==o['top10_asins'][:3] for o in orders)
    for row in counts:
        candidate,tied,strict,multi=[int(row[k]) for k in ('candidate_users','same_bucket_target_users','strict_panel_users','multi_event_last_bucket_users')]
        assert math.isclose(float(row['same_bucket_target_percent']),100*tied/candidate,abs_tol=1e-12)
        assert math.isclose(float(row['multi_event_last_bucket_percent']),100*multi/strict,abs_tol=1e-12)
        assert sum(int(row['bucket'+k+'_users']) for k in ('2','3','4','5plus'))==multi
        for k in ('2','3','4','5plus'):assert math.isclose(float(row['bucket'+k+'_within_multi_event_percent']),100*int(row['bucket'+k+'_users'])/multi,abs_tol=1e-12)
    return case,counts


def canvas(height):
    fig=style.figure(height);ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,style.WIDTH_MM);ax.set_ylim(0,height);ax.axis('off')
    return fig,ax


def text(ax,x,y,value,**kwargs):
    defaults=dict(fontsize=style.FONT,fontfamily='Arial',color=GRAY,ha='left',va='center')
    defaults.update(kwargs);return ax.text(x,y,value,**defaults)


def rule(ax,x0,x1,y,color=style.RULE,width=.5):ax.plot([x0,x1],[y,y],color=color,lw=width,solid_capstyle='butt',clip_on=False)


def box(ax,x0,y0,w,h,fc,ec,lw=.6,ls='-',z=1):
    ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0,rounding_size=1.0',fc=fc,ec=ec,lw=lw,ls=ls,zorder=z))


def text_width(ax,artist):
    renderer=ax.figure.canvas.get_renderer();bb=artist.get_window_extent(renderer)
    (x0,_),(x1,_)=ax.transData.inverted().transform([[bb.x0,bb.y0],[bb.x1,bb.y1]]);return x1-x0


def chip(ax,x,y,label,fc='white',ec=style.PREFIX_EDGE,pad=1.3,h=4.8,**kwargs):
    """Draw a rounded label whose left edge is x; return its right edge."""
    artist=text(ax,x+pad,y,label,zorder=4,**kwargs);w=text_width(ax,artist)+2*pad
    box(ax,x,y-h/2,w,h,fc,ec,lw=.5,z=3);return x+w


def arrow(ax,x0,x1,y,color=style.REF):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle='-|>',mutation_scale=7,lw=.7,color=color,shrinkA=0,shrinkB=0,zorder=2))


def fig1(case):
    fig,ax=canvas(HEIGHTS['Fig1'])
    fig.canvas.draw()
    label_of=lambda asin:case['labels'][asin]['short_display_label']
    target=label_of(case['target_asin'])
    # Observed record: fixed prefix, unordered same-day bucket, strictly later target.
    ytop,yrow=62.5,55.5
    text(ax,2,ytop,'History prefix',fontweight='bold')
    box(ax,2,yrow-4.5,30,9,PREFIX_FILL,PREFIX_EDGE)
    text(ax,17,yrow,'21 events, fixed',ha='center')
    arrow(ax,33,38.5,yrow)
    title=text(ax,39.5,ytop,r'Final bucket $B_m$: 17 Apr 2014',fontweight='bold')
    text(ax,39.5+text_width(ax,title)+1.2,ytop,'(order not recorded)',color=MUTED,fontsize=7.5)
    bucket_items=[label_of(a) for a in case['orders'][0]['last_bucket_asins']]
    right=chip(ax,42.5,yrow,bucket_items[0]);right=chip(ax,right+2.5,yrow,bucket_items[1])
    box(ax,39.5,yrow-4.5,right+3-39.5,9,BUCKET_FILL,BUCKET_EDGE,z=1)
    arrow(ax,right+4,right+9.5,yrow)
    tx=right+10.5
    text(ax,tx,ytop,'Target: 25 Jun 2014',fontweight='bold')
    chip(ax,tx,yrow,target,fc=TARGET_FILL,ec=TARGET_EDGE)
    rule(ax,1,173,47.5)
    ax.plot([87,87],[8,45],color=style.RULE,lw=.45)
    displayed=[]
    for index,order in enumerate(case['orders']):
        x=2+index*87.5
        text(ax,x,43,'('+('a' if index==0 else 'b')+') Legal order '+order['label'],fontweight='bold')
        labels=[label_of(asin) for asin in order['last_bucket_asins']]
        text(ax,x,36.5,'Input',color=MUTED,fontsize=7.5)
        r=chip(ax,x+10,36.5,labels[0],fc=BUCKET_FILL,ec=BUCKET_EDGE);arrow(ax,r+1,r+4.5,36.5);chip(ax,r+5.5,36.5,labels[1],fc=BUCKET_FILL,ec=BUCKET_EDGE)
        text(ax,x,29.5,'Top 3',color=MUTED,fontsize=7.5)
        for rank,asin in enumerate(order['display_top3_asins'],1):
            is_target=asin==case['target_asin'];label=label_of(asin);y=29.5-(rank-1)*6
            if is_target:box(ax,x+9,y-2.5,52,5,TARGET_FILL,TARGET_EDGE,lw=.5)
            text(ax,x+11,y,str(rank),fontweight='bold');text(ax,x+15.5,y,label)
            displayed.append({'order':order['label'],'display_rank':rank,'asin':asin,'short_display_label':label,'source_title':case['labels'][asin]['source_title'],'target':is_target})
        text(ax,x+73,29.5,'Target rank',ha='center',color=MUTED,fontsize=7.5)
        text(ax,x+73,21,str(order['target_rank']),ha='center',fontsize=17,fontweight='bold')
    text(ax,87,3.2,'Full Top-10 lists share 0 of 10 items    |    Shaded row: target item',ha='center',color=MUTED)
    return fig,{'body_top_k':3,'full_overlap_top_k':10,'full_top10_intersection':0,'target_ranks':[1,335],'internal_permutation_ids_visible':False,'product_display_records':displayed}


def fig2(case):
    fig,ax=canvas(HEIGHTS['Fig2'])
    fig.canvas.draw()
    # (a) The strict-target rule and the legal test views on one time axis.
    text(ax,2,84,'(a) Strict target and legal test views',fontweight='bold')
    ax.add_patch(plt.Rectangle((50,58),68,23,fc=style.TIE_BG,ec='none',zorder=0))
    text(ax,84,78.8,'same recorded time bucket',ha='center',color=MUTED,fontsize=7.5)
    yb=68
    for x0,label in ((4,r'$B_1$'),(26,r'$B_{m-1}$')):
        box(ax,x0,yb-4,14,8,PREFIX_FILL,PREFIX_EDGE);text(ax,x0+7,yb,label,ha='center')
    text(ax,22,yb,r'$\cdots$',ha='center')
    ax.plot([4,40],[61.6,61.6],color=MUTED,lw=.5);ax.plot([4,4],[61.6,62.6],color=MUTED,lw=.5);ax.plot([40,40],[61.6,62.6],color=MUTED,lw=.5)
    text(ax,22,59.3,r'fixed prefix $P$',ha='center',color=MUTED,fontsize=7.5)
    box(ax,52,yb-4.5,38,9,BUCKET_FILL,BUCKET_EDGE)
    text(ax,54,yb+6.3,r'final history bucket $B_m$',fontsize=7.5,color=MUTED)
    r=56
    for item in 'abc':r=chip(ax,r,yb,'$'+item+'$',pad=1.6)+3
    text(ax,71,59.3,r'test view: $P\oplus\pi(B_m)$, five fixed $\pi$',ha='center',color=MUTED,fontsize=7.5)
    box(ax,99,yb-4.5,15,9,'white',EXCLUDED_EDGE,ls=(0,(3,2)))
    text(ax,106.5,yb,r'$x$',ha='center',color=EXCLUDED_EDGE)
    text(ax,106.5,yb+6.3,'tied target',ha='center',fontsize=7.5,color=EXCLUDED_EDGE)
    text(ax,106.5,yb-7.2,'excluded',ha='center',fontsize=7.5,color=EXCLUDED_EDGE)
    box(ax,138,yb-4.5,15,9,TARGET_FILL,TARGET_EDGE);text(ax,145.5,yb,r'$y$',ha='center')
    text(ax,145.5,yb+6.3,'strict target',ha='center',fontsize=7.5,color=MUTED)
    text(ax,145.5,yb-7.2,r'$b(y)>b(B_m)$',ha='center',fontsize=7.5,color=MUTED)
    arrow(ax,2,172,55.5);text(ax,172,53.2,r'recorded time bucket $b(\cdot)$',ha='right',color=MUTED,fontsize=7.5)
    # (b) Where each method intervenes.
    text(ax,2,48.5,'(b) Where each method intervenes',fontweight='bold')
    text(ax,2,43,'Method',fontweight='bold');text(ax,32,43,'Training stage',fontweight='bold');text(ax,106,43,'Inference stage',fontweight='bold')
    rule(ax,1,173,40.5,color=GRAY,width=.7)
    rows=[('Control',('Reference order','Own fitted checkpoint'),('One legal view',)),
          ('Shuffle',('All visible tied buckets shuffled','Own fitted checkpoint'),('One legal view',)),
          ('Ensemble',('Reuse fitted Control checkpoint','No additional training'),('Five legal views','Mean raw scores (before mask)')),
          ('LBSP',(r'Mean-pool $B_m$, then positions','Own fitted checkpoint'),(r'Mean-pool $B_m$, then positions','One output per legal view'))]
    for i,(method,train,infer) in enumerate(rows):
        y=36-i*8.5;style_row=style.METHODS[method]
        ax.scatter([3],[y],s=19,marker=style_row['marker'],color=style_row['color'],linewidths=.6)
        text(ax,7,y,method,color=style_row['color'],fontweight='bold')
        for j,label in enumerate(train):text(ax,32,y+1.8-j*3.8,label)
        if len(infer)==1:text(ax,106,y,infer[0])
        else:
            for j,label in enumerate(infer):text(ax,106,y+1.8-j*3.8,label)
        if i<3:rule(ax,1,173,y-4.25,color=style.GRID,width=.45)
    rule(ax,1,173,6.2,color=GRAY,width=.7)
    text(ax,87,2.5,'All methods: raw scores → mask from uncompressed history → rank / Top-10',ha='center',color=MUTED)
    return fig,{'protocol_contract':case['protocol_contract'],'shared_workflow_drawn_once':True,'Control_and_Ensemble_share_checkpoint':True,'Shuffle_and_LBSP_separately_trained':True,'Ensemble_mean_before_mask':True,'common_bucket_notation':'B_m','large_rounded_frames':False,'method_styles':style.METHODS}


def fig3(counts):
    fig,ax=canvas(HEIGHTS['Fig3'])
    text(ax,24,59,'(a) Tied-target exclusions',fontweight='bold')
    text(ax,96,59,'(b) Last-bucket composition',fontweight='bold')
    text(ax,84,51,'Candidate N',ha='right',color=MUTED)
    text(ax,165,51,'Multi / strict',ha='center',color=MUTED)
    text(ax,96,51.4,'Size',color=MUTED)
    for i,(label,color) in enumerate(zip(('2','3','4','≥5'),style.BUCKET_COLORS)):
        x=104+i*11
        ax.add_patch(plt.Rectangle((x,50),3,2.8,facecolor=color,edgecolor='none'))
        text(ax,x+4,51.4,label)
    left_x,left_w=24,40
    right_x,right_w=96,50
    ybase=7.0
    for value in (0,25,50):
        x=left_x+left_w*value/50;ax.plot([x,x],[ybase,46],lw=.4,color=style.GRID,zorder=0);text(ax,x,4.7,str(value),ha='center')
    for value in (0,25,50,75,100):
        x=right_x+right_w*value/100;ax.plot([x,x],[ybase,46],lw=.4,color=style.GRID,zorder=0);text(ax,x,4.7,str(value),ha='center')
    for i,row in enumerate(counts):
        y=42-i*10
        label=style.DATASET_LABELS[row['dataset']]
        text(ax,1,y,label)
        value=float(row['same_bucket_target_percent'])
        ax.add_patch(plt.Rectangle((left_x,y-2.3),left_w*value/50,4.6,facecolor=style.TIE_RAMP[3],edgecolor='none'))
        text(ax,left_x+left_w*value/50+1.2,y,f'{value:.1f}%')
        text(ax,84,y,f'{int(row["candidate_users"]):,}',ha='right')
        cursor=right_x
        for k,(key,color) in enumerate(zip(('2','3','4','5plus'),style.BUCKET_COLORS)):
            share=float(row['bucket'+key+'_within_multi_event_percent']);width=right_w*share/100
            ax.add_patch(plt.Rectangle((cursor,y-2.3),width,4.6,facecolor=color,edgecolor='white',linewidth=.35))
            if share>=10:text(ax,cursor+width/2,y,f'{share:.0f}',ha='center',fontsize=6.5,color='white' if k>=2 else GRAY)
            cursor+=width
        assert math.isclose(cursor,right_x+right_w,abs_tol=1e-10)
        text(ax,165,y+1.5,f'{float(row["multi_event_last_bucket_percent"]):.1f}%',ha='center')
        text(ax,165,y-2.2,f'{int(row["multi_event_last_bucket_users"]):,}/{int(row["strict_panel_users"]):,}',ha='center')
    rule(ax,left_x,left_x+left_w,ybase,color=GRAY,width=.6);rule(ax,right_x,right_x+right_w,ybase,color=GRAY,width=.6)
    text(ax,44,1.6,'Excluded / candidate users (%)',ha='center')
    text(ax,121,1.6,'Within multi-event users (%)',ha='center')
    return fig,{'numeric_rows':counts,'left_denominator':'candidate_users_before_other_strict_eligibility_exclusions','right_bar_denominator':'multi_event_last_bucket_users','right_annotation_denominator':'strict_panel_users','bucket_order':['2','3','4','>=5'],'bucket_colors':style.BUCKET_COLORS,'shared_dataset_rows':True,'strict_panel_not_assumed_candidate_minus_ties':True,'invented_seed_error_bars':False}


def save_figure(fig,name,out):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.canvas.get_width_height();bounds=[]
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():continue
        box=artist.get_window_extent(renderer)
        bounds.append({'text':artist.get_text(),'font_pt':artist.get_fontsize(),'pixel_bbox':[float(box.x0),float(box.y0),float(box.x1),float(box.y1)]})
        assert box.x0>=-.5 and box.y0>=-.5 and box.x1<=W+.5 and box.y1<=H+.5,(name,artist.get_text(),box,W,H)
    preview=out/'previews';preview.mkdir(exist_ok=True)
    fig.savefig(preview/(name+'.png'),dpi=240,bbox_inches=None,pad_inches=0)
    style.save(fig,out/(name+'.pdf'))
    reader=PdfReader(out/(name+'.pdf'));assert len(reader.pages)==1
    page=reader.pages[0];assert not page.images
    dimensions=[float(page.mediabox.width)*25.4/72,float(page.mediabox.height)*25.4/72]
    assert math.isclose(dimensions[0],174,abs_tol=1e-4) and math.isclose(dimensions[1],HEIGHTS[name],abs_tol=1e-4)
    return {'width_mm':dimensions[0],'height_mm':dimensions[1],'body_font_pt':style.FONT,'vector_raster_image_count':0,'pdf_sha256':sha(out/(name+'.pdf')),'preview_png_sha256':sha(preview/(name+'.png')),'text_geometry':bounds}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=HERE);parser.add_argument('--font-dir',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    font=font_setup(args.font_dir);case,counts=inputs();results={}
    for name,(figure,detail) in zip(('Fig1','Fig2','Fig3'),(fig1(case),fig2(case),fig3(counts))):
        results[name]=dict(save_figure(figure,name,out),**detail)
    audit={'status':'BUILT_VECTOR_GEOMETRY_PASS_VISUAL_REVIEW_PENDING','date':'2026-10-31','portable':True,'runtime_data_dependencies':list(INPUT_SHA),'input_sha256':{n:sha(INPUTS/n) for n in INPUT_SHA},
           'original_source_records':case['source_records'],'shared_style_sha256':sha(HERE/'paper_style.py'),'builder_sha256':sha(Path(__file__)),'font':font,'figures':results,
           'statistics_or_experimental_data_changed':False,'raw_interaction_checkpoint_or_research_code_accessed':False,'project_manuscript_or_canonical_changed':False,
           'caption_facts':{'Fig1':'Only Top3 product labels shown; target ranks1and335 and zero FULLTop10intersection remain unchanged. Product names are abbreviated; no internalpermutation IDs or gold shading are displayed. Saved hitflipcase is selected, not random.',
                            'Fig2':'Shared strict target/history/candidate workflow precedes4methodrows. Shuffle changesallvisibletrainingtiedbuckets; Ensemble reusesControlcheckpoint andaverages5raw-scorevectors beforecommonmask/rank; LBSPpoolsB_mbeforepositions. FiveEnsembleoutputgroups arein thecaption, not5newmodels.',
                            'Fig3':'Panel(a) candidate-user denominator beforeothereligibilityfilters; panel(b) conditionalwithinmulti-event strict histories. Rightannotations=incidence%andmulti/strictcounts. Singletons excludedfromcomposition; sharedrowdatasets; fixedcountdescriptorswithoutCI.'}}
    (out/'structural_figures_audit.json').write_bytes((json.dumps(audit,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'status':audit['status'],'dimensions_mm':{k:[v['width_mm'],v['height_mm']] for k,v in results.items()},'output_dir':str(out)},indent=2))


if __name__=='__main__':main()
