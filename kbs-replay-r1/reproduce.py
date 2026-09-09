"""Portable fixed-record scheduling reproduction and bounded post-hoc sensitivity.

No model, physics, recovery, MTD, or anomaly generator is imported. Input records
are evaluator-owned. Policy admission/ranking only sees a positive forecast list.
Outputs must be new; --out will never replace an existing run.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import sys
sys.dont_write_bytecode=True
import argparse, dataclasses, types, json, csv, contextlib, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import milp
from safe_numeric_io import load_numeric,save_numeric_new,write_json_new,sha256_file
import policies_p02_copy as engine
import planning_replay as planner
from reference_audit import reference_audit
R=Path(__file__).resolve().parent
METHODS=('S0','A_Q25_FIFO','ALL_ACCEPT_FIFO','ATC_START','ROLLING_CURRENT_QUEUE_NONDELAY')
VISIBLE=('job_id','arrival_step','verify_score','ddd_loss','pred_attack_prob','pred_service_time','pred_service_cost','pred_busy_steps','pred_fail_prob','pred_attack_severity','pred_expected_consequence','value_proxy')
PARAMS=('v_weight','clean_penalty','busy_penalty','admission_score_threshold')
_score,_admit=engine._policy_score,engine._admission_accept
def project(job,now):
    if job.arrival_step>now:raise ValueError('future arrival')
    return types.SimpleNamespace(**{k:getattr(job,k) for k in VISIBLE})
def bind_forecast_only():
    def score(item,**kw):
        kw['active_servers']=tuple(None for _ in kw['active_servers'])
        return _score(types.SimpleNamespace(job=project(item.job,kw['step']),enqueue_step=item.enqueue_step),**kw)
    def admit(job,**kw):
        kw['active_servers']=tuple(None for _ in kw['active_servers'])
        return _admit(project(job,kw['step']),**kw)
    engine._policy_score=score;engine._admission_accept=admit
def validate_jobs(j,T):
    required={f.name for f in dataclasses.fields(engine.AlarmJob) if f.name!='meta'}
    if set(j)!=required:raise ValueError('missing/extra job fields')
    n=len(j['job_id'])
    if any(v.shape!=(n,) or not np.isfinite(v).all() for v in j.values()):raise ValueError('shape or finite contract')
    if not np.array_equal(j['job_id'],np.arange(n)):raise ValueError('job identity')
    a=j['arrival_step']
    if not np.all((a>=0)&(a<T)&(a==np.floor(a))) or np.any(np.diff(a)<0):raise ValueError('arrival contract')
    for k in ('is_attack','actual_backend_fail'):
        if not np.isin(j[k],[0,1]).all():raise ValueError('binary outcome required')
    for k in ('actual_busy_steps','pred_busy_steps'):
        if np.any(j[k]<1) or np.any(j[k]!=np.floor(j[k])):raise ValueError('duration contract')
    for k in ('pred_attack_prob','pred_fail_prob'):
        if not np.all((j[k]>=0)&(j[k]<=1)):raise ValueError('probability contract')
    if np.any(j['severity_true']<0):raise ValueError('negative severity')
def validate_config(c):
    if c['window_cost_budget'] is not None or c['cost_budget_window_steps']!=0:raise ValueError('actual-cost admission path prohibited')
    for k in ('fail_penalty','urgency_bonus','cost_penalty','adaptive_gain'):
        if c[k]!=0:raise ValueError('outside bounded audit support')
    if c['max_wait_steps']!=10 or c['slot_budget'] not in (1,2):raise ValueError('unsupported config')
def metrics_equal(new,old):
    keys=('R_weighted_nonfail','R_count_nonfail','R_count','starts','clean_starts','busy_steps','occupied_in_horizon','failures','recorded_cost','D95','expired','horizon','unfinished_T')
    return {k:{'new':new[k],'old':old[k]} for k in keys if not (new[k]==old[k] or (new[k] is not None and old[k] is not None and np.isclose(new[k],old[k],atol=1e-10,rtol=1e-12)))}
def simple(j,T,c,method,gate):
    c=dict(c);validate_config(c)
    family='S';spec=None
    if method=='A_Q25_FIFO':
        c.update(policy_name='threshold_expected_consequence_fifo',threshold=gate);family='A';spec={'id':'Q25','value':gate}
    if method=='ALL_ACCEPT_FIFO':
        c.update(policy_name='fifo',threshold=None);family='A';spec={'id':'ALL_ACCEPT','value':None}
    cfg=engine.SimulationConfig(**c)
    jobs=[engine.AlarmJob(**{k:v[i].item() for k,v in j.items()}) for i in range(len(j['job_id']))]
    result=engine.simulate_policy(jobs,total_steps=T,cfg=cfg)
    audit=reference_audit(jobs,T,cfg,family,spec,result)
    return dict(config=c,result=result,metrics=audit['metrics'],start_step=audit['start_step'],terminal=audit['status'],audit=audit)
def write_csv(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    conf=json.loads((R/'data/CONFIG.json').read_bytes())
    if sha256_file(R/'data/BANK.npz')!=conf['bank_sha256'] or sha256_file(R/'data/HISTORICAL_REFERENCE.npz')!=conf['gold_sha256']:raise ValueError('input identity')
    bank=load_numeric(R/'data/BANK.npz');gold=load_numeric(R/'data/HISTORICAL_REFERENCE.npz');T=conf['horizon'];gate=conf['A_Q25_gate']
    for tag in conf['tags']:
        validate_jobs(bank[tag]['jobs_evaluator_only'],T)
        for B in (1,2):validate_config(conf['configs'][tag][str(B)])
    write_json_new(out/'RUN_CONTEXT.json',dict(scope_sha256=sha256_file(R/'ANALYSIS_SCOPE.md'),python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,bank_sha256=conf['bank_sha256'],forecast_policy_fields=VISIBLE,maximum_trajectories=224,physical_calls=0,production_model_calls=0,new_input_generation=0,role=conf['role']))
    bind_forecast_only();planner.milp=milp
    baseline={};rows=[];comparisons=[];sens=[];counter={'A3':0};trajectories=0
    with (out/'MILP_CALLS.jsonl').open('x') as journal,(out/'EXECUTION.log').open('x') as log:
        for tag in conf['tags']:
            j=bank[tag]['jobs_evaluator_only']
            for B in (1,2):
                for method in METHODS:
                    key=f'{tag}__{method}__B{B}'
                    if method in METHODS[:3]:
                        v=simple(j,T,conf['configs'][tag][str(B)],method,gate)
                        save_numeric_new(out/'baseline'/(key+'.npz'),v)
                    else:
                        with contextlib.redirect_stdout(log):
                            receipt=planner.run(j,T,B,method,gate,tag,'A3',counter,journal,output_dir=out)
                        v=load_numeric(receipt['path'])
                    trajectories+=1;baseline[key]=v;m=v['metrics'];g=gold[key]
                    comparison=dict(tag=tag,B=B,method=method,starts_equal=bool(np.array_equal(v['start_step'],g['start_step'])),terminal_equal=bool(np.array_equal(v['terminal'],g['terminal'])),metric_mismatches=metrics_equal(m,g['metrics']))
                    comparisons.append(comparison)
                    row=dict(tag=tag,B=B,method=method,**{k:m[k] for k in ('R_weighted_nonfail','R_count_nonfail','starts','clean_starts','busy_steps','occupied_in_horizon','recorded_cost','D95','failures','expired','horizon')},MILP=m.get('MILP',0),fallbacks=m.get('fallbacks',0))
                    rows.append(row)
                print('BASELINE_COMPLETE',tag,'B'+str(B),flush=True)
        # Retain every outcome even if reproduction reveals a platform discrepancy.
        write_json_new(out/'HISTORICAL_COMPARISON.json',comparisons)
        if not all(x['starts_equal'] and x['terminal_equal'] and not x['metric_mismatches'] for x in comparisons):
            write_csv(out/'BASELINE_METRICS.csv',rows)
            raise RuntimeError('Baseline discrepancy; no sensitivity or scientific conclusion until inspected. Raw run preserved.')
        for tag in conf['tags']:
            j=bank[tag]['jobs_evaluator_only']
            for B in (1,2):
                base=baseline[f'{tag}__S0__B{B}'];c0=conf['configs'][tag][str(B)]
                for param in (*PARAMS,*(('age_bonus',) if B==1 else ())):
                    for factor in (0.8,1.2):
                        c=dict(c0);c[param]*=factor;v=simple(j,T,c,'S0',gate);trajectories+=1
                        key=f'{tag}__B{B}__{param}__{factor}'
                        save_numeric_new(out/'sensitivity'/(key+'.npz'),v)
                        a=v['start_step']>=0;b=base['start_step']>=0;m=v['metrics'];bm=base['metrics']
                        sens.append(dict(tag=tag,B=B,parameter=param,factor=factor,base_value=c0[param],new_value=c[param],**{k:m[k] for k in ('R_weighted_nonfail','R_count_nonfail','starts','clean_starts','busy_steps','recorded_cost','D95')},delta_Rw_pp=100*(m['R_weighted_nonfail']-bm['R_weighted_nonfail']),delta_Rn_pp=100*(m['R_count_nonfail']-bm['R_count_nonfail']),delta_busy=m['busy_steps']-bm['busy_steps'],delta_starts=m['starts']-bm['starts'],selected_symmetric_difference=int(np.sum(a!=b)),selected_jaccard=float(np.sum(a&b)/np.sum(a|b))))
            print('SENSITIVITY_COMPLETE',tag,flush=True)
    write_csv(out/'BASELINE_METRICS.csv',rows);write_csv(out/'SENSITIVITY_ALL.csv',sens)
    write_json_new(out/'EXECUTION_COMPLETE.json',dict(trajectories=trajectories,timesteps=trajectories*T,MILP=counter['A3'],all_80_baseline_actions_and_metrics_match=True,sensitivity_traces=len(sens),physical_calls=0,production_model_calls=0,new_input_generation=0))
    print('COMPLETE',trajectories,'traces',counter,'no physical or production-model calls',flush=True)
if __name__=='__main__':main()
