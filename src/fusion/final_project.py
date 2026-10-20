from pathlib import Path
import sys, json, csv, html
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance, ImageFilter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.radar.reader import load_dataset, read_radar_sample
from src.radar.filters import filter_radar_points
from src.radar.clustering import build_candidates
from src.radar.audit import get_radar_boxes, audit_candidates
from src.radar.labels import create_candidate_labels
from src.camera.reader import read_camera_sample
from src.camera.detector import load_detector, detect_objects
from src.fusion.projection import project_radar_to_image
from src.fusion.association import associate_points
from src.fusion.decision import load_radar_model, predict_radar_candidates, build_fusion_decisions
from src.fusion.final_class import finalize_fusion

def main():
    out=ROOT/'final_results'
    out.mkdir(exist_ok=True)
    source=ROOT/'src/radar/results/final_training_balanced_random_forest_2026-10-13_11-32-39'
    metadata=json.loads((source/'training_metadata.json').read_text())
    model=load_radar_model(source/'evaluation_model.joblib')
    dataset=load_dataset(ROOT/'data/downloads/v1.0-mini')
    detector=load_detector()
    rows=[]; details=[]; gallery=[]
    scenarios=['original','dark','blur','camera_unavailable']
    for scene in dataset.scene:
        if scene['name'] not in metadata['reserved_scenes']: continue
        tokens=[]; token=scene['first_sample_token']
        while token:
            tokens.append(token); token=dataset.get('sample',token)['next']
        for position in np.linspace(0,len(tokens)-1,5,dtype=int):
            token=tokens[position]
            radar=read_radar_sample(dataset,token)
            camera=read_camera_sample(dataset,token)
            filtered=filter_radar_points(radar['points'])
            candidates=build_candidates(filtered)
            predictions=predict_radar_candidates(model,candidates)
            audit=audit_candidates(filtered,candidates,get_radar_boxes(dataset,radar['radar_token']))
            labels={x['candidate_id']:x['label'] for x in create_candidate_labels(candidates,audit)}
            projection=project_radar_to_image(radar,camera)
            for scenario in scenarios:
                image=Image.fromarray(camera['image'])
                if scenario=='dark': image=ImageEnhance.Brightness(image).enhance(.2)
                if scenario=='blur': image=image.filter(ImageFilter.GaussianBlur(5))
                pixels=np.array(image)
                detections=[] if scenario=='camera_unavailable' else detect_objects(detector,pixels)
                association=associate_points(detections,projection)
                decisions=finalize_fusion(build_fusion_decisions(predictions,association))['decisions']
                scores={name:Counter() for name in ['radar','camera','fusion']}
                for d in decisions:
                    reference=labels.get(d['candidate_id'])
                    if reference is None: continue
                    matches=d['camera_matches']
                    supported=[m for m in matches if m['camera_class'] is not None]
                    camera_class=max(supported,key=lambda m:m['camera_score'])['camera_class'] if supported else None
                    for name,pred in [('radar',d['radar_class']),('camera',camera_class),('fusion',d['final_class'])]:
                        scores[name]['total']+=1
                        available=pred not in [None,'uncertain']
                        scores[name]['available']+=available
                        scores[name]['correct']+=available and pred==reference
                        details.append(dict(scene=scene['name'],sample=token,scenario=scenario,candidate=d['candidate_id'],reference=reference,sensor=name,prediction=pred,kind=d.get("candidate_kind","unknown")))
                for sensor,counts in scores.items():
                    rows.append(dict(scene=scene['name'],sample=token,scenario=scenario,sensor=sensor,**counts))
                fig,axes=plt.subplots(1,3,figsize=(22,7))
                for candidate in candidates:
                    p=candidate['points']; axes[0].scatter(p[:,0],p[:,1],s=12)
                axes[0].set(title='Radar XY candidates',xlabel='X (m)',ylabel='Y (m)'); axes[0].axis('equal')
                for ax in axes[1:]: ax.imshow(pixels); ax.axis('off')
                axes[1].set_title('Camera detections' if scenario!='camera_unavailable' else 'Camera unavailable (reference image)')
                axes[2].set_title('Fusion: candidate-level final classes')
                for detection in detections:
                    x,y,x2,y2=detection['box']
                    linked=[d for d in decisions if any(m['detection_id']==detection['detection_id'] for m in d['camera_matches'])]
                    final=dict(Counter(d['final_class'] for d in linked))
                    for ax,text in [(axes[1],f"{detection['project_class']} {detection['confidence']:.2f}"),(axes[2],str(final))]:
                        ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor='yellow'))
                        ax.text(x,y,text,color='yellow',fontsize=7,bbox=dict(facecolor='black',alpha=.6))
                uv=projection['pixels']
                axes[2].scatter(uv[:,0],uv[:,1],c=projection['radar_ranges'],s=10,cmap='viridis')
                name=f"{scene['name']}_{position}_{scenario}.png"
                fig.suptitle(f"{scene['name']} sample {position} — {scenario}")
                fig.tight_layout(); fig.savefig(out/name,dpi=120); plt.close(fig)
                gallery.append(name)
                print(scene['name'],position,scenario,flush=True)
    for name,data in [('metrics.csv',rows),('predictions.csv',details)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    aggregates=[]
    for scenario in scenarios:
        for sensor in ['radar','camera','fusion']:
            selected=[r for r in rows if r['scenario']==scenario and r['sensor']==sensor]
            totals={k:sum(r.get(k,0) for r in selected) for k in ['total','available','correct']}
            n=totals['total']; a=totals['available']; c=totals['correct']
            aggregates.append(dict(scenario=scenario,sensor=sensor,**totals,coverage=100*a/n if n else 0,correct_over_all=100*c/n if n else 0,accuracy_when_available=100*c/a if a else None))
    (out/'summary.json').write_text(json.dumps(aggregates,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for sensor in ['radar','camera','fusion']:
        subset=[r for r in aggregates if r['sensor']==sensor]
        for ax,key in zip(axes,['correct_over_all','coverage']):
            values=[r[key] for r in subset]; ax.plot(scenarios,values,'o-',label=sensor)
            for x,y in zip(scenarios,values): ax.annotate(f'{y:.1f}%',(x,y),fontsize=8)
            ax.set_ylim(0,105); ax.set_ylabel(key+' (%)'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=150); plt.close(fig)
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in r.values())+'</tr>' for r in aggregates)
    headers=''.join(f'<th>{k}</th>' for k in aggregates[0])
    report='''<html><meta charset="utf-8"><style>body{font:17px Arial;margin:40px;background:#fafafa}img{max-width:100%}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Radar–camera MVP: comparative experiment</h1><p>This experiment tests complementarity; it does not assume fusion wins. Ten samples from two scenes excluded from evaluation-model training are used. These scenes were previously inspected, so this is exploratory validation.</p><p>Darkening and Gaussian blur are controlled image perturbations, not simulated physical rain. Radar remains unchanged. Camera-unavailable tests sensor outage. Actual radar degradation and adverse-weather robustness are not established.</p><p>Reference labels are provisional radar XY associations. Scores compare classifications on the same labeled radar candidates. Camera results are conditional associations, not detection precision or mAP. Candidates and frames are correlated. A box can influence multiple candidates. Abstentions count against correct-over-all; conditional accuracy excludes abstentions.</p><h2>Comparison</h2><img src="comparison.png"><table><tr>'''+headers+'</tr>'+table+'</table><h2>Visual diagnostics</h2>'+''.join(f'<h3>{name}</h3><img loading="lazy" src="{name}">' for name in gallery)+'</html>'
    from src.fusion.result_reporting import enrich_report
    report=enrich_report(out, report, details, aggregates)
    (out/'report.html').write_text(report,encoding='utf-8')
    print('RESULTS:',out,flush=True)

if __name__=='__main__': main()
