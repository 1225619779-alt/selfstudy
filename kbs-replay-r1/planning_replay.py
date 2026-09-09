"""P05C scheduling functions copied verbatim. New portable I/O entry only."""
from pathlib import Path
import json,time,warnings,dataclasses
import numpy as np
from safe_numeric_io import save_numeric_new,write_json_new,sha256_file
from p05b_scheduler_core import observation_from_visible,atc_order,rolling_plan
def decide(obs,method,solve):
    if method=='ATC_START':return dict(start_now=list(atc_order(obs)),reason='atc',fallback=False,solver_called=False)
    jobs=[j for j in obs.waiting if j.predicted_value>0 and j.latest_start>=obs.now]
    if jobs and len(jobs)<=obs.free_slots:
        return dict(start_now=[j.job_id for j in jobs],reason='no_choice_all_start',fallback=False,solver_called=False)
    return rolling_plan(obs,solver=solve)

def check(j,T,B,gate,trace,starts,terminal):
    a=j['arrival_step'];n=len(a);ids=np.flatnonzero(starts>=0);ends=starts+j['actual_busy_steps']
    assert len(trace)==T and np.all(terminal>=0)
    assert np.array_equal(terminal==1,j['pred_expected_consequence']<gate)
    assert np.all((starts[ids]>=a[ids])&(starts[ids]<=a[ids]+10))
    assert np.all((a[terminal==2]+11)<T) and np.all((a[terminal==3]+11)>=T)
    for t,row in enumerate(trace):
        active=np.flatnonzero((starts>=0)&(starts<t)&(ends>t))
        waiting=np.flatnonzero((terminal!=1)&(a<=t)&(a+10>=t)&((starts<0)|(starts>=t)))
        actual=np.flatnonzero(starts==t)
        assert set(row['decision']['start_now'])==set(actual.tolist())
        assert set(row['visible_waiting_ids'])==set(waiting.tolist())
        assert set(row['visible_active_ids'])==set(active.tolist())
        assert len(actual)==min(B-len(active),len(waiting))
        assert len(active)+len(actual)<=B
        assert row['after_active']==len(active)+len(actual)
    y=j['is_attack']==1;f=j['actual_backend_fail']!=0;s=j['severity_true'];ok=(starts>=0)&y&~f
    ratio=lambda x,d:float(x/d) if d else None
    return dict(R_weighted_nonfail=ratio(s[ok].sum(),s[y].sum()),R_count_nonfail=ratio(ok.sum(),y.sum()),R_count=ratio(((starts>=0)&y).sum(),y.sum()),denominator=float(s[y].sum()),numerator=float(s[ok].sum()),starts=len(ids),clean_starts=int(((starts>=0)&~y).sum()),busy_steps=int(j['actual_busy_steps'][ids].sum()),occupied_in_horizon=int((np.minimum(ends[ids],T)-starts[ids]).sum()),arrival_rejected=int((terminal==1).sum()),expired=int((terminal==2).sum()),horizon=int((terminal==3).sum()),failures=int(f[ids].sum()),recorded_cost=float(j['actual_service_cost'][ids].sum()),D95=float(np.quantile(starts[ids]-a[ids],.95)) if len(ids) else None,unfinished_T=int(((starts>=0)&(ends>T)).sum()))

def run(j,T,B,method,gate,tag,stage,counter,journal,*,output_dir):
    n=len(j['job_id']);assert np.array_equal(j['job_id'],np.arange(n))
    assert np.all(j['actual_busy_steps']>=1)
    out=Path(output_dir)/stage/f'{tag}__{method}__B{B}.npz'
    write_json_new(out.with_suffix('.started.json'),dict(tag=tag,method=method,B=B,T=T,stage=stage))
    starts=np.full(n,-1,dtype=np.int64);terminal=np.full(n,-1,dtype=np.int8)
    active=[];waiting=[];trace=[];tic=time.perf_counter();solves=0;solver_seconds=0.;decision_seconds=0.
    for now in range(T):
        active=[i for i in active if starts[i]+j['actual_busy_steps'][i]>now]
        for i in np.flatnonzero(j['arrival_step']==now):
            i=int(i)
            if j['pred_expected_consequence'][i]>=gate:waiting.append(i)
            else:terminal[i]=1
        for i in list(waiting):
            if now>j['arrival_step'][i]+10:waiting.remove(i);terminal[i]=2
        obs=observation_from_visible(now=now,capacity=B,waiting=[dict(job_id=i,arrival=int(j['arrival_step'][i]),latest_start=int(j['arrival_step'][i])+10,predicted_value=float(j['pred_expected_consequence'][i]),predicted_duration=float(j['pred_busy_steps'][i])) for i in waiting],active=[dict(job_id=i,started_at=int(starts[i]),predicted_duration=float(j['pred_busy_steps'][i])) for i in active])
        records=[]
        def solve(**kwargs):
            nonlocal solves,solver_seconds
            cap=64 if stage=='A2' else 8640
            if counter[stage]>=cap:raise SystemExit('Hard solve cap reached')
            counter[stage]+=1;solves+=1
            journal.write(json.dumps(dict(event='before',stage=stage,tag=tag,B=B,now=now,index=counter[stage]))+'\n');journal.flush()
            t=time.perf_counter();rec={}
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');res=milp(**kwargs)
                rec=dict(status=int(res.status),message=str(res.message),x=res.x,gap=getattr(res,'mip_gap',None),warnings=[str(w.message) for w in ws])
                return res
            except Exception as e:rec=dict(exception=type(e).__name__,message=str(e));raise
            finally:
                seconds=time.perf_counter()-t;solver_seconds+=seconds;rec['seconds']=seconds;records.append(rec)
                journal.write(json.dumps(dict(event='after',stage=stage,index=counter[stage],seconds=seconds,status=rec.get('status'),exception=rec.get('exception')))+'\n');journal.flush()
        t=time.perf_counter();d=decide(obs,method,solve);decision_seconds+=time.perf_counter()-t
        ids=d['start_now'];assert len(ids)==len(set(ids)) and len(ids)<=B-len(active) and all(i in waiting for i in ids)
        wids=list(waiting);aids=list(active)
        for i in ids:waiting.remove(i);active.append(i);starts[i]=now;terminal[i]=0
        trace.append(dict(now=now,decision=d,solver=records,visible_waiting_ids=wids,visible_active_ids=aids,after_active=len(active)))
    for i in waiting:terminal[i]=3
    metrics=check(j,T,B,gate,trace,starts,terminal)
    metrics.update(MILP=solves,fallbacks=sum(bool(r['decision']['fallback']) for r in trace),solver_limit_status=sum(s.get('status')==1 for r in trace for s in r['solver']),no_choice=sum(r['decision']['reason']=='no_choice_all_start' for r in trace),solver_seconds=solver_seconds,decision_including_solver_seconds=decision_seconds,model_and_decision_excluding_solver_seconds=max(0,decision_seconds-solver_seconds),replay_and_check_seconds=time.perf_counter()-tic)
    payload=dict(tag=tag,method=method,B=B,T=T,gate=gate,role='ENGINEERING_EXPOSED_DEV_SMOKE' if stage=='A2' else 'EXPOSED_RETROSPECTIVE_SHARED_INFORMATION_COMPARISON',jobs_evaluator_only=j,start_step=starts,terminal=terminal,trace=trace,metrics=metrics,checks='PASS_CAPACITY_DEADLINE_RELEASE_PARTITION',predictor_DTO_fields=[f.name for f in dataclasses.fields(obs.waiting[0])] if obs.waiting else ['job_id','arrival','latest_start','predicted_value','predicted_duration'])
    t=time.perf_counter();save_numeric_new(out,payload)
    receipt=dict(tag=tag,method=method,B=B,stage=stage,metrics=metrics,save_seconds=time.perf_counter()-t,path=str(out),sha256=sha256_file(out))
    print(json.dumps(receipt),flush=True);return receipt
