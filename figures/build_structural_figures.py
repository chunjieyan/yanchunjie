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
import matplotlib.pyplot as plt
from pypdf import PdfReader
import paper_style as style

HERE=Path(__file__).resolve().parent
INPUTS=HERE/'inputs'
INPUT_SHA={'fig1_example.json':'9a00b484bbf59e6a6efbbb9c65b203b5515dc0c152f58af6f2510a3c980e0341',
           'fig3_counts.csv':'a63d04dc6bbf8b8b13a94c4af79851bc1f2b155b35eda90c511ae6278135dbda'}
HEIGHTS={'Fig1':65,'Fig2':78,'Fig3':62}
GRAY=style.METHODS['Control']['color']


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


def rule(ax,x0,x1,y,color='#858585',width=.5):ax.plot([x0,x1],[y,y],color=color,lw=width,solid_capstyle='butt',clip_on=False)


def fig1(case):
    fig,ax=canvas(HEIGHTS['Fig1'])
    text(ax,87,61,'Same user, fitted Control model, target and candidate set',ha='center',fontweight='bold')
    text(ax,87,56,'Fixed prefix: 21 events    |    Final bucket $B_m$: 17 Apr 2014 (UTC)',ha='center')
    target=case['labels'][case['target_asin']]['short_display_label']
    text(ax,87,51,'Target: '+target+' (25 Jun 2014)',ha='center')
    rule(ax,1,173,47.5)
    ax.plot([87,87],[9,45],color='#B5B5B5',lw=.45)
    displayed=[]
    for index,order in enumerate(case['orders']):
        x=2+index*88
        text(ax,x,44,'('+('a' if index==0 else 'b')+') Legal order '+order['label'],fontweight='bold')
        labels=[case['labels'][asin]['short_display_label'] for asin in order['last_bucket_asins']]
        # Both strings are verified abbreviated product labels, not fabricated names.
        text(ax,x,38.5,labels[0]+' → '+labels[1])
        text(ax,x,32.5,'Top 3 recommendations shown',fontweight='bold')
        for rank,asin in enumerate(order['display_top3_asins'],1):
            is_target=asin==case['target_asin'];label=case['labels'][asin]['short_display_label']
            text(ax,x,27-(rank-1)*5,f'{rank}  '+label,fontweight='bold' if is_target else 'normal')
            displayed.append({'order':order['label'],'display_rank':rank,'asin':asin,'short_display_label':label,'source_title':case['labels'][asin]['source_title'],'target':is_target})
        text(ax,x,10.5,'Target rank: '+str(order['target_rank']),fontweight='bold')
    text(ax,87,3.2,'Full Top-10 lists share 0 items    |    Target item in bold',ha='center',fontweight='bold')
    return fig,{'body_top_k':3,'full_overlap_top_k':10,'full_top10_intersection':0,'target_ranks':[1,335],'internal_permutation_ids_visible':False,'product_display_records':displayed}


def fig2(case):
    fig,ax=canvas(HEIGHTS['Fig2'])
    text(ax,87,74,r'Strict target: $b(y)>b(B_m)$; exclude $x$ if $b(x)=b(B_m)$',ha='center',fontweight='bold')
    text(ax,87,68,r'Fixed prefix $[B_1,\ldots,B_{m-1}]$  →  $B_m=\{a,b,c\}$  →  strictly later $y$',ha='center')
    text(ax,87,62,r'Test view: fixed prefix + $\pi(B_m)$; target, history multiset and candidates fixed',ha='center')
    text(ax,2,56.5,'Method',fontweight='bold');text(ax,32,56.5,'Training stage',fontweight='bold');text(ax,106,56.5,'Inference stage',fontweight='bold')
    rule(ax,1,173,54,color=GRAY,width=.7)
    rows=[('Control',('Reference order','Own fitted checkpoint'),('One legal view',)),
          ('Shuffle',('All visible tied buckets shuffled','Own fitted checkpoint'),('One legal view',)),
          ('Ensemble',('Reuse fitted Control checkpoint','No additional training'),('Five legal views','Mean raw scores (before mask)')),
          ('LBSP',(r'Mean-pool $B_m$, then positions','Own fitted checkpoint'),(r'Mean-pool $B_m$, then positions','One output per legal view'))]
    for i,(method,train,infer) in enumerate(rows):
        y=46.5-i*11;style_row=style.METHODS[method]
        ax.scatter([3],[y],s=19,marker=style_row['marker'],color=style_row['color'],linewidths=.6)
        text(ax,7,y,method,color=style_row['color'],fontweight='bold')
        for j,label in enumerate(train):text(ax,32,y+1.8-j*3.8,label)
        if len(infer)==1:text(ax,106,y,infer[0])
        else:
            for j,label in enumerate(infer):text(ax,106,y+1.8-j*3.8,label)
        if i<3:rule(ax,1,173,y-5.5,color='#B8B8B8',width=.35)
    rule(ax,1,173,6.7,color=GRAY,width=.7)
    text(ax,87,2.5,'Raw scores → mask from uncompressed history → rank / Top-10',ha='center')
    return fig,{'protocol_contract':case['protocol_contract'],'shared_workflow_drawn_once':True,'Control_and_Ensemble_share_checkpoint':True,'Shuffle_and_LBSP_separately_trained':True,'Ensemble_mean_before_mask':True,'common_bucket_notation':'B_m','large_rounded_frames':False,'method_styles':style.METHODS}


def fig3(counts):
    fig,ax=canvas(HEIGHTS['Fig3'])
    text(ax,24,59,'(a) Tied-target exclusions',fontweight='bold')
    text(ax,96,59,'(b) Last-bucket composition',fontweight='bold')
    text(ax,84,51,'Candidate N',ha='right')
    text(ax,165,51,'Multi / strict',ha='center')
    for i,(label,color) in enumerate(zip(('2','3','4','≥5'),style.BUCKET_COLORS)):
        x=97+i*12
        ax.add_patch(plt.Rectangle((x,50),3,2.8,facecolor=color,edgecolor='none'))
        text(ax,x+4,51.4,label)
    left_x,left_w=24,40
    right_x,right_w=96,50
    ybase=7.0
    for value in (0,25,50):
        x=left_x+left_w*value/50;ax.plot([x,x],[ybase,46],lw=.3,color='#D2D2D2',zorder=0);text(ax,x,4.7,str(value),ha='center')
    for value in (0,25,50,75,100):
        x=right_x+right_w*value/100;ax.plot([x,x],[ybase,46],lw=.3,color='#D2D2D2',zorder=0);text(ax,x,4.7,str(value),ha='center')
    for i,row in enumerate(counts):
        y=42-i*10
        label=style.DATASET_LABELS[row['dataset']]
        text(ax,1,y,label)
        value=float(row['same_bucket_target_percent'])
        ax.add_patch(plt.Rectangle((left_x,y-2.3),left_w*value/50,4.6,facecolor=GRAY,edgecolor='none'))
        text(ax,left_x+left_w*value/50+1.2,y,f'{value:.1f}%')
        text(ax,84,y,f'{int(row["candidate_users"]):,}',ha='right')
        cursor=right_x
        for key,color in zip(('2','3','4','5plus'),style.BUCKET_COLORS):
            width=right_w*float(row['bucket'+key+'_within_multi_event_percent'])/100
            ax.add_patch(plt.Rectangle((cursor,y-2.3),width,4.6,facecolor=color,edgecolor='white',linewidth=.35));cursor+=width
        assert math.isclose(cursor,right_x+right_w,abs_tol=1e-10)
        text(ax,165,y+1.5,f'{float(row["multi_event_last_bucket_percent"]):.1f}%',ha='center')
        text(ax,165,y-2.2,f'{int(row["multi_event_last_bucket_users"]):,}/{int(row["strict_panel_users"]):,}',ha='center')
    rule(ax,left_x,left_x+left_w,ybase,width=.5);rule(ax,right_x,right_x+right_w,ybase,width=.5)
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
