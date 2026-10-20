from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import csv

CLASSES=['vehicle','pedestrian','two_wheeler','obstacle']

def enrich_report(out, report, details, aggregates):
    sections=['<h2>Selection policy and interpretation</h2><p>The camera category takes priority, including conflicts. Multiple boxes are resolved by the highest camera score among supported categories. Radar supplies a fallback when no supported camera prediction exists. Uncertainty remains only when neither source supplies a category. This policy increases coverage but may propagate camera errors. Both camera-only and fusion use the same box-selection rule for a fair comparison.</p>']
    for scenario in sorted({r['scenario'] for r in aggregates}):
        subset=[r for r in aggregates if r['scenario']==scenario]
        best=max(r['correct_over_all'] for r in subset)
        winners=', '.join(r['sensor'] for r in subset if r['correct_over_all']==best)
        sections.append(f'<h2>{scenario}: best correct-over-all = {winners} ({best:.2f}%)</h2><p>The winner maximizes correct classifications across all labeled candidates, including cases without an answer. This is not object detection recall.</p>')
        for sensor in ['radar','camera','fusion']:
            selected=[r for r in details if r['scenario']==scenario and r['sensor']==sensor]
            matrix=np.zeros((4,5),dtype=int)
            for row in selected:
                i=CLASSES.index(row['reference']); pred=row['prediction']
                j=CLASSES.index(pred) if pred in CLASSES else 4
                matrix[i,j]+=1
            fig,ax=plt.subplots(figsize=(9,5));ax.imshow(matrix,cmap='Blues')
            ax.set_xticks(range(5),CLASSES+['no answer'],rotation=25);ax.set_yticks(range(4),CLASSES)
            ax.set(xlabel='Prediction',ylabel='Reference',title=f'{scenario} — {sensor}: counts and row percentages')
            for i in range(4):
                for j in range(5):
                    n=matrix[i].sum(); ax.text(j,i,f'{matrix[i,j]}\n{100*matrix[i,j]/n:.1f}%' if n else f'{matrix[i,j]}\nN/A',ha='center',va='center',color='red')
            fig.tight_layout();filename=f'confusion_{scenario}_{sensor}.png';fig.savefig(out/filename,dpi=130);plt.close(fig)
            table=[]
            for i,name in enumerate(CLASSES):
                tp=matrix[i,i];predicted=matrix[:,i].sum();actual=matrix[i].sum()
                precision=f'{tp}/{predicted} = {100*tp/predicted:.2f}%' if predicted else 'N/A (no predictions)'
                recall=f'{tp}/{actual} = {100*tp/actual:.2f}%' if actual else 'N/A (no references)'
                table.append([name,precision,recall])
            fig,ax=plt.subplots(figsize=(12,3));ax.axis('off');ax.set_title(f'{scenario} — {sensor}: per-class precision and recall')
            t=ax.table(cellText=table,colLabels=['Class','Precision: TP / predicted','Recall: TP / reference'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
            metricfile=f'metrics_{scenario}_{sensor}.png';fig.tight_layout();fig.savefig(out/metricfile,dpi=140);plt.close(fig)
            sections.append(f'<h3>{sensor}</h3><img src="{filename}"><img src="{metricfile}">')
    sections.append('<h2>Radar clustering analysis</h2><p>Clusters and isolated returns are evaluated on original images only to avoid counting the same radar predictions four times. These are different populations: a better cluster result is evidence of association with performance, not a causal ablation of DBSCAN.</p>')
    for kind in ['cluster','isolated_return']:
        rows=[r for r in details if r['scenario']=='original' and r['sensor']=='radar' and r['kind']==kind]
        correct=sum(r['reference']==r['prediction'] for r in rows)
        sections.append(f'<p><b>{kind}</b>: {correct}/{len(rows)} correct ({100*correct/len(rows):.2f}%).</p>' if rows else f'<p>{kind}: no examples.</p>')
    sections.append('<p>Distance, radial structure and multiple returns can supply useful radar features. However, the present model is weak on several categories. Camera outage demonstrates fallback capability; improvement under real rain requires separate data. No thresholds were selected to force fusion to win.</p>')
    return report.replace('<h2>Visual diagnostics</h2>',''.join(sections)+'<h2>Visual diagnostics</h2>')
