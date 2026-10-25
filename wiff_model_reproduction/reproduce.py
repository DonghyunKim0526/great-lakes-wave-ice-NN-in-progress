"""Run archived-weight diagnostics, fresh SP-WIFF comparisons, and save figures.

Run `python fetch_data.py` first. No TensorFlow or MATLAB is required.
"""
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'data/mplconfig'))
import argparse,csv,hashlib,json,platform,sys,time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from threadpoolctl import threadpool_limits,threadpool_info
from wiff import ROOT,DenseNetwork,SPWIFF,parameters,histogram_metrics,eligible
from fetch_data import digest,MD5

FIGURES=ROOT/'figures';RESULTS=ROOT/'results'
COLORS={'truth':'#273B4A','nn':'#D95B43','single':'#2C94A1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#273B4A','axes.titlesize':11,'figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42})


def save(fig,name,subtitle):
    fig.suptitle(subtitle,fontsize=13,fontweight='bold',color='#17384D')
    fig.text(.01,-.035,'WIFF v1.0b | generated from numerical outputs | archived-data diagnostics are not the original held-out split',fontsize=7,color='#526271')
    fig.savefig(FIGURES/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(FIGURES/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def sample_archive(path,count,seed):
    """Uniform row sample without loading the entire uncompressed archive."""
    cache=ROOT/'data'/f'sample-{count}-seed{seed}.npz'
    if cache.exists():
        with np.load(cache) as d:return d['inputs'],d['targets'],d['row_ids'],int(d['population'])
    with h5py.File(path,'r') as handle:
        a,b=handle['in'],handle['out']
        # MATLAB v7.3 stores a variable's dimensions in reverse order.
        if a.shape[0] not in (27,28,29,30) or b.shape[0]!=12 or a.shape[1]!=b.shape[1]:
            raise ValueError(f'Unexpected dataset shapes: {a.shape}, {b.shape}')
        population=a.shape[1]
        ids=np.sort(np.random.default_rng(seed).choice(population,min(count,population),replace=False))
        x=np.empty((len(ids),27));y=np.empty((len(ids),12))
        for start in range(0,population,32768):
            stop=min(population,start+32768)
            left,right=np.searchsorted(ids,[start,stop])
            if left==right:continue
            within=ids[left:right]-start
            x[left:right]=a[:27,start:stop].T[within]
            y[left:right]=b[:,start:stop].T[within]
        np.savez_compressed(cache,inputs=x,targets=y,row_ids=ids,population=population)
    return x,y,ids,population


def binned(values,metric,edges):
    loc=np.digitize(values,edges)-1
    centers=(edges[:-1]+edges[1:])/2
    stats=np.full((3,len(centers)),np.nan)
    counts=np.zeros(len(centers))
    for j in range(len(centers)):
        subset=metric[loc==j];subset=subset[np.isfinite(subset)];counts[j]=len(subset)
        if len(subset)>=20:stats[:,j]=np.percentile(subset,[25,50,75])
    return centers,stats,counts


def evaluate(inputs,targets,row_ids,population,seed):
    histogram=DenseNetwork();classifier=DenseNetwork('classifier')
    f,df,edges,centers=parameters()
    keep=eligible(inputs)
    x,y,ids=inputs[keep],targets[keep],row_ids[keep]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Archive contains invalid values')
    fracture=y.sum(axis=1)>0
    if not np.allclose(y[fracture].sum(axis=1),1,atol=1e-5):
        raise ValueError('Fracture labels are not normalized bin masses')
    p=classifier.predict(x)[:,1]
    call=p>=.54
    prediction=histogram.predict(x[fracture])
    scores=histogram_metrics(y[fracture],prediction,centers)
    energy=np.maximum(x[:,:25]@df,1e-300)
    fp=(~fracture)&call;fn=fracture&(~call)
    tp=fracture&call;tn=(~fracture)&(~call)
    metrics={
        'sample_seed':seed,'archive_population':int(population),'uniform_sample_size':len(inputs),'eligible_sample_size':len(x),'fracturing_sample_size':int(fracture.sum()),
        'evaluation_scope':'Archived-data diagnostic using pretrained weights. The original train/validation membership is unavailable; this is NOT a held-out test.',
        'threshold':.54,'classifier_probability':'second softmax component = P(fracture), following training labels and archived Fortran',
        'classifier_accuracy':float(np.mean(call==fracture)),
        'false_positive_rate':float(fp.sum()/(~fracture).sum()),'false_negative_rate':float(fn.sum()/fracture.sum()),
        'confusion_matrix_true_rows_predicted_columns':[[int(tn.sum()),int(fp.sum())],[int(fn.sum()),int(tp.sum())]],
        'histogram_median_sse_percent':float(100*np.median(scores['sse'])),
        'histogram_mean_rse_m':float(np.mean(scores['rse_m'])),
        'histogram_mean_sae':float(np.mean(scores['sae'])),
        'histogram_median_symmetric_radius_error_percent':float(100*np.median(scores['symmetric_radius_error'])),
        'paper_heldout_median_sse_percent_reference':3.9,
        'fraction_normalized_predictions':float(np.mean(np.isclose(prediction.sum(axis=1),1))),
    }
    # Figure 1: classifier conditional behavior and confusion matrix.
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    axes_vars=[(x[:,26],np.linspace(0,1,19),'Ice concentration'),(x[:,25],np.linspace(0,3,19),'Ice thickness (m)'),(np.log10(energy),np.linspace(-3.3,1,20),'log10 spectral variance (m²)')]
    for ax,(v,bins,label) in zip(axs.flat,axes_vars):
        for values,color,name in [(call,'#273B4A','Run fraction'),(call==fracture,'#2C94A1','Accuracy'),(fp,'#D95B43','False positive / all'),(fn,'#8C6BB1','False negative / all')]:
            xx,st,_=binned(v,values.astype(float),bins)
            # Rates are means, not medians of binary values.
            index=np.digitize(v,bins)-1
            yy=[values[index==j].mean() if np.sum(index==j)>=20 else np.nan for j in range(len(xx))]
            ax.plot(xx,yy,color=color,label=name)
        ax.set(xlabel=label,ylabel='Fraction',ylim=(0,1));ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    cm=np.array(metrics['confusion_matrix_true_rows_predicted_columns'])
    ax=axs[1,1];ax.imshow(cm,cmap='Blues')
    for (i,j),v in np.ndenumerate(cm):ax.text(j,i,f'{v:,}',ha='center',va='center',color='white' if v>cm.max()/2 else '#17384D',fontsize=15)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No fracture','Fracture'],yticklabels=['No fracture','Fracture'],xlabel='Predicted',ylabel='Archived target',title=f'Accuracy {metrics["classifier_accuracy"]:.2%} | threshold 0.54')
    save(fig,'01_classifier_diagnostics','Classifier diagnostics on the official archived dataset')

    # Figure 2: conditional RSE/R plus probability-mass histogram comparison.
    fig=plt.figure(figsize=(11,10),layout='constrained');gs=fig.add_gridspec(3,2)
    variables=[(x[fracture,26],np.linspace(0,1,21),'Ice concentration'),(x[fracture,25],np.linspace(0,3,21),'Ice thickness (m)'),(np.log10(energy[fracture]),np.linspace(-3.3,1,22),'log10 spectral variance (m²)'),(scores['radius'],np.geomspace(centers[0],centers[-1],22),'Reference representative radius (m)')]
    for i,(v,bins,label) in enumerate(variables):
        ax=fig.add_subplot(gs[i//2,i%2]);xx,st,n=binned(v,100*scores['sse'],bins)
        twin=ax.twinx();twin.bar(xx,n/n.sum(),width=np.diff(bins)*.9,color='#E6B5A8',alpha=.5);twin.set_ylabel('Sample fraction',color='#A56A58')
        ax.set_zorder(2);ax.patch.set_visible(False);ax.plot(xx,st[1],color=COLORS['truth']);ax.fill_between(xx,st[0],st[2],color=COLORS['truth'],alpha=.15)
        ax.set(xlabel=label,ylabel='SSE = RSE / R (%)',ylim=(0,min(100,float(np.nanpercentile(st[2],95))*1.2+1)));ax.grid(alpha=.2)
        if i==3:ax.set_xscale('log');twin.set_xscale('log')
    ax=fig.add_subplot(gs[2,:]);ax.semilogx(centers,y[fracture].mean(axis=0),'-o',color=COLORS['truth'],label='Archived SP-WIFF label');ax.semilogx(centers,prediction.mean(axis=0),'-s',color=COLORS['nn'],label='Archived NN-WIFF weights')
    ax.set(xlabel='Floe radius bin center (m)',ylabel='Mean bin probability mass');ax.grid(alpha=.2);ax.legend()
    save(fig,'02_histogram_diagnostics',f'Histogram diagnostics | median SSE {metrics["histogram_median_sse_percent"]:.2f}% (not held out)')

    # Figure 3: all six phase-space pairs, with explicit low-occupancy masking.
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    variables=[(x[fracture,26],'Ice concentration',np.linspace(0,1,18)),(x[fracture,25],'Ice thickness (m)',np.linspace(0,3,18)),(np.log10(energy[fracture]),'log10 variance (m²)',np.linspace(-3.3,1,18)),(np.log10(scores['radius']),'log10 radius (m)',np.linspace(np.log10(centers[0]),np.log10(centers[-1]),18))]
    for ax,(i,j) in zip(axs.flat,[(0,2),(1,2),(3,2),(0,1),(1,3),(0,3)]):
        vx,lx,bx=variables[i];vy,ly,by=variables[j]
        med,_,_,_=binned_statistic_2d(vx,vy,100*scores['sse'],statistic='median',bins=[bx,by])
        counts,_,_=np.histogram2d(vx,vy,bins=[bx,by]);med[counts<20]=np.nan
        m=ax.pcolormesh(bx,by,med.T,cmap='viridis',vmin=0,vmax=50,shading='flat')
        ax.set(xlabel=lx,ylabel=ly)
    fig.colorbar(m,ax=axs,label='Median SSE (%) | cells with <20 samples masked',shrink=.8,extend='max')
    save(fig,'03_error_phase_space','Conditional histogram error on archived fracture states')

    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    quantiles=np.linspace(.001,.999,999)
    for key,label,color in [('sse','Weighted histogram error: paper Eq. 6-9',COLORS['truth']),('symmetric_radius_error','Symmetric mean-radius error: released plotting helper',COLORS['nn'])]:
        values=100*scores[key]
        ax.semilogx(np.maximum(np.quantile(values,quantiles),1e-6),quantiles,color=color,label=f'{label}\nMedian = {np.median(values):.2f}%')
    ax.axhline(.5,color='#8B9AA3',linestyle=':',linewidth=1)
    ax.set(xlabel='Error (%) - definitions are different',ylabel='Empirical cumulative fraction',ylim=(0,1))
    ax.legend(fontsize=9,loc='lower right');ax.grid(alpha=.2)
    save(fig,'07_metric_definition_comparison','Why the released plotting metric differs from the paper equations')

    # Keep derived evidence small enough to inspect directly in the repository.
    with (RESULTS/'archive_diagnostics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['archive_row','ice_thickness_m','ice_concentration','variance_m2','true_fracture','predicted_fracture','p_fracture'])
        for i in range(len(x)):writer.writerow([int(ids[i]),*x[i,25:27],energy[i],int(fracture[i]),int(call[i]),p[i]])
    np.savez_compressed(ROOT/'data/archive_predictions.npz',inputs=x,targets=y,row_ids=ids,fracture_mask=fracture,predictions=prediction)
    return metrics,x[fracture],y[fracture],prediction,ids[fracture]


def numerical_comparisons(inputs,targets,predicted,row_ids,ncases,seed):
    f,df,edges,centers=parameters();kernel=SPWIFF();network=DenseNetwork()
    chosen=np.random.default_rng(seed+1).choice(len(inputs),min(ncases,len(inputs)),replace=False)
    records=[];truths=[];single=[];nn=[];times=[];runtimes=[]
    for number,index in enumerate(chosen):
        x=inputs[index]
        t=time.perf_counter();result=kernel.run(x[:25],x[25],seed=seed+number,max_iterations=300);elapsed=time.perf_counter()-t
        one=kernel.run(x[:25],x[25],max_iterations=1,phases=np.full(25,np.pi))
        prediction=network.predict(x)[0]
        # Repeated, warmed one-state inference timings, excluding disk loading.
        latency=[]
        for _ in range(9):
            start=time.perf_counter();network.predict(x);latency.append(time.perf_counter()-start)
        metric=histogram_metrics(result.histogram,prediction,centers)
        records.append({'archive_row':int(row_ids[index]),'iterations':result.iterations,'converged':result.converged,'sp_seconds':elapsed,'nn_single_seconds':float(np.median(latency)),'fresh_sp_histogram_sum':float(result.histogram.sum()),'nn_vs_fresh_sp_sse_percent':float(100*metric['sse']) if np.isfinite(metric['sse']) else None,'nn_vs_archive_sse_percent':float(100*histogram_metrics(targets[index],prediction,centers)['sse'])})
        truths.append(result.histogram);single.append(one.histogram);nn.append(prediction);times.append(float(np.median(latency)));runtimes.append(elapsed)
        print(f'SP case {number+1}/{len(chosen)}: {result.iterations} iterations, {elapsed:.3f}s',flush=True)
    truths,single,nn=map(np.array,[truths,single,nn])
    summary={'scope':'Fresh random-phase MATLAB stand-alone port at archived physical states; not independent observations or the original CICE Fortran trajectory.',
             'cases':len(chosen),'converged_cases':sum(r['converged'] for r in records),'median_sp_seconds_per_state':float(np.median(runtimes)),
             'median_nn_single_seconds_per_state':float(np.median(times)),
             'ratio_of_median_component_latencies':float(np.median(runtimes)/np.median(times)),
             'median_nn_vs_fresh_sp_sse_percent':float(np.nanmedian(100*histogram_metrics(truths,nn,centers)['sse'])),
             'median_single_iteration_vs_fresh_sp_sse_percent':float(np.nanmedian(100*histogram_metrics(truths,single,centers)['sse'])),
             'timing_note':'Single CPU BLAS thread; loaded weights and precomputed Fourier basis excluded; classifier/CICE/I-O excluded. Python component timings are not the paper\'s CICE speedup.'}
    with (RESULTS/'fresh_sp_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    np.savez_compressed(RESULTS/'fresh_sp_histograms.npz',inputs=inputs[chosen],archive_rows=row_ids[chosen],archive_labels=targets[chosen],sp_histograms=truths,single_iteration_histograms=single,nn_histograms=nn)
    fig,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,j in zip(axs.flat,range(min(6,len(chosen)))):
        for values,color,label in [(truths[j],COLORS['truth'],'Fresh SP (seeded)'),(nn[j],COLORS['nn'],'NN-WIFF'),(single[j],COLORS['single'],'Fixed-phase 1 iter')]:ax.semilogx(centers,values,'-o',markersize=3,color=color,label=label)
        x=inputs[chosen[j]];ax.set(title=f'Archive row {row_ids[chosen[j]]:,}\nH={x[25]:.2f}m, C={x[26]:.2f}',xlabel='Floe radius (m)',ylabel='Bin probability mass');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    save(fig,'04_fresh_sp_comparison','Fresh numerical fracture histograms versus pretrained NN-WIFF')
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    axs[0].scatter(runtimes,np.asarray(times)*1000,color=COLORS['nn']);axs[0].set(xlabel='SP kernel time per state (s)',ylabel='NN histogram time per state (ms)',title='Paired component latency');axs[0].grid(alpha=.2)
    axs[1].bar(['SP kernel','NN histogram'],[np.median(runtimes)*1000,np.median(times)*1000],color=[COLORS['truth'],COLORS['nn']]);axs[1].set_yscale('log');axs[1].set(ylabel='Median component latency (ms, log scale)',title='No CICE, classifier, or wave-model runtime')
    save(fig,'05_component_runtime','Measured component runtime on this computer')
    return summary


def convergence_example(seed):
    kernel=SPWIFF();network=DenseNetwork();_,_,_,centers=parameters()
    # Exactly the physical input distributed in SP_WIFF_Standalone.m.
    from re import search
    original=(ROOT/'vendor/wiff-v1.0b/Model/SP_WIFF_Standalone.m').read_text()
    spectrum=np.fromstring(search(r'WSPEC\s*=\s*\[([^]]+)\]',original).group(1),sep=',')
    thickness=float(search(r'\nH\s*=\s*([^;]+)',original).group(1));concentration=float(search(r'\nC\s*=\s*([^;]+)',original).group(1))
    fig,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');runs=[]
    for offset in range(5):
        result=kernel.run(spectrum,thickness,seed=seed+offset,max_iterations=1000)
        runs.append({'seed':seed+offset,'iterations':result.iterations,'converged':result.converged,'last_mean_histogram_change':float(result.changes[-1])})
        axs[0].semilogy(np.arange(1,result.iterations+1),np.maximum(result.changes,1e-12),label=f'Seed {seed+offset}')
        axs[1].semilogx(centers,result.histogram,alpha=.7,label=f'SP seed {offset+1}')
    axs[0].axhline(6.5e-4,color=COLORS['nn'],linestyle='--',label='Released-code tolerance');axs[0].set(xlabel='Random-phase iterations',ylabel='Mean absolute histogram update',title='Stopping rule from the released MATLAB code');axs[0].legend(fontsize=7);axs[0].grid(alpha=.2)
    prediction=network.predict(np.r_[spectrum,thickness,concentration])[0]
    axs[1].semilogx(centers,prediction,color=COLORS['nn'],linewidth=2.3,label='NN-WIFF');axs[1].set(xlabel='Floe radius (m)',ylabel='Bin probability mass',title='Same input; different random-phase histories');axs[1].legend(fontsize=7);axs[1].grid(alpha=.2)
    save(fig,'06_stochastic_convergence','Released stand-alone example: convergence and random-phase sensitivity')
    (RESULTS/'convergence_example.json').write_text(json.dumps({'input':np.r_[spectrum,thickness,concentration].tolist(),'runs':runs},indent=2))
    return runs


def write_result_report(archive,numerical):
    text=f'''# Executed WIFF1.0 component reproduction

## Dataset and numerical results

The official archive contains **{archive['archive_population']:,} states**. This run selects **{archive['uniform_sample_size']:,} rows uniformly without replacement**, seed **{archive['sample_seed']}**. All {archive['eligible_sample_size']:,} selected rows pass the paper's thickness, concentration, and wave-height screen; {archive['fracturing_sample_size']:,} have nonzero reference fracture histograms.

| Diagnostic | Measured result | Interpretation |
| --- | ---: | --- |
| Classifier accuracy | {archive['classifier_accuracy']:.3%} | Archived pretrained weights; paper reports about 87.5% |
| False-positive rate | {archive['false_positive_rate']:.3%} | False calls divided by reference nonfracture states |
| False-negative rate | {archive['false_negative_rate']:.3%} | Missed calls divided by reference fracture states |
| Mean representative size error | {archive['histogram_mean_rse_m']:.3f} m | Weighted absolute histogram error; paper gives about 30 m |
| Median RSE / R | {archive['histogram_median_sse_percent']:.3f}% | Paper Eq. 6-9 definition, computed on archived fracture states |
| Median symmetric mean-radius error | {archive['histogram_median_symmetric_radius_error_percent']:.3f}% | Released plotting-helper definition; numerically close to reported 3.9% |
| Normalized NN histograms | {archive['fraction_normalized_predictions']:.1%} | All evaluated probability vectors sum to one |

**This is an archived-data diagnostic, not an independent held-out score.** Original split membership is not available, and the pretrained weights may have been trained on sampled rows. Numerical proximity to a paper statistic does not establish reproduction of the original validation split.

## Main finding: the two error definitions are not interchangeable

The paper defines `RSE = sum(r_i * abs(q_pred_i - q_true_i))` and `SSE = RSE / R_true`. The released `Paper_Figures/Misc/training_preamble.m` instead defines `perc_error = 100 * abs(R_true - R_pred) / (0.5 * (R_true + R_pred))`. The latter allows errors in different bins to cancel when forming the mean radius; the former does not.

On exactly the same {archive['fracturing_sample_size']:,} states, the medians are **{archive['histogram_median_sse_percent']:.2f}%** and **{archive['histogram_median_symmetric_radius_error_percent']:.2f}%**, respectively. We preserve both definitions and do not relabel the smaller number as weighted histogram error. This source-code/equation discrepancy is material to interpreting a reproduction of the reported 3.9% result. The [metric-comparison figure](figures/07_metric_definition_comparison.png) makes it visible. This finding concerns these source definitions and this run; it is not a claim that the original held-out outputs have been independently audited.

## Fresh numerical kernel comparisons

- Randomly selected physical states: **{numerical['cases']}**; numerical stopping criterion met in **{numerical['converged_cases']}**.
- Median NN-versus-fresh-SP weighted histogram error: **{numerical['median_nn_vs_fresh_sp_sse_percent']:.2f}%**.
- Median fixed-phase single-iteration-versus-fresh-SP weighted histogram error: **{numerical['median_single_iteration_vs_fresh_sp_sse_percent']:.2f}%**.
- Median Python SP kernel latency: **{1000*numerical['median_sp_seconds_per_state']:.3f} ms/state**.
- Median warmed NN histogram latency: **{1000*numerical['median_nn_single_seconds_per_state']:.4f} ms/state**.
- Ratio of those median component latencies: **{numerical['ratio_of_median_component_latencies']:.1f}x**.

These {numerical['cases']} states and newly generated phase realizations do not establish generalization to new climates or observations. The SP kernel is the documented MATLAB stand-alone port, not a recompiled CICE Fortran kernel. In this small comparison, the error values above must be retained even if the neural model does not outperform a one-iteration reference. The speed ratio is a component benchmark on this computer, not an end-to-end CICE speedup. Fourier-basis construction, disk loading, classification, I/O, and wave-model execution are excluded; one BLAS thread is used.

## Reproducibility and evidence

- [metrics.json](results/metrics.json): full numerical results and execution environment.
- [source_provenance.json](results/source_provenance.json): official data/code URLs, checksums, and sizes.
- [archive_diagnostics.csv](results/archive_diagnostics.csv): all sampled row IDs and classifier outcomes; row IDs are zero-based in the archive.
- [fresh_sp_cases.csv](results/fresh_sp_cases.csv): all selected cases, timing, stopping status, and errors.
- [fresh_sp_histograms.npz](results/fresh_sp_histograms.npz): selected inputs, archived labels, new numerical histograms, and NN outputs.
- [convergence_example.json](results/convergence_example.json): exact published stand-alone example input and five seeded stopping histories.
- `python -m pytest -q`: tests of text/HDF5 weight parity, row-wise softmax, literal MATLAB peak/curvature logic, physical metrics, seeds, and upstream file checksums.

The raw 753 MB data remain outside Git tracking. No Great Lakes adaptation, new network training, global CICE rerun, or paper Figures 4-5 reproduction is claimed.
'''
    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size',type=int,default=100000)
    parser.add_argument('--sp-cases',type=int,default=24)
    parser.add_argument('--seed',type=int,default=20261024)
    args=parser.parse_args()
    if args.sample_size<100 or args.sp_cases<1:parser.error('Need sample-size >=100 and sp-cases >=1')
    FIGURES.mkdir(exist_ok=True);RESULTS.mkdir(exist_ok=True)
    path=ROOT/'data/Training_converged.mat'
    if not path.exists():raise SystemExit('Run python fetch_data.py first')
    if digest(path)!=MD5:raise SystemExit('Official training-data MD5 mismatch')
    with threadpool_limits(limits=1):
        x,y,ids,population=sample_archive(path,args.sample_size,args.seed)
        print('Archive sample loaded:',x.shape,y.shape,flush=True)
        metrics,xf,yf,pred,fracture_ids=evaluate(x,y,ids,population,args.seed)
        print('Archive diagnostics:',json.dumps(metrics),flush=True)
        numerical=numerical_comparisons(xf,yf,pred,fracture_ids,args.sp_cases,args.seed)
        convergence=convergence_example(args.seed)
        environment={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'h5py':h5py.__version__,'threadpools':threadpool_info(),'blas_thread_limit':1}
    result={'archive_diagnostics':metrics,'fresh_sp_comparison':numerical,'convergence':convergence,'environment':environment}
    (RESULTS/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_result_report(metrics,numerical)
    print('Finished. Figures:',FIGURES,flush=True)


if __name__=='__main__':main()
