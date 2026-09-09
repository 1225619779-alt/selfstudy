"""P02 independent action audit; five fixed coefficients parameterized for P09. No action generation."""
import dataclasses,math
import numpy as np
UNIT=9.574143552780152
def reference_audit(jobs,T,cfg,family,spec,result):
 """Verify saved actions by independent per-step arithmetic; not a policy replay.
 Reconstruct each predecision state from recorded starts, admission rule and TTL.
 Exact ID/order comparison proves state transitions inductively without generating actions.
 """
 n=len(jobs); ids=np.asarray(result['served_jobs'],int)
 steps=np.asarray([int(t['step']) for t in result['step_trace'] for _ in range(int(t['selected_count']))],int)
 assert len(ids)==len(steps) and len(set(ids.tolist()))==len(ids)
 fields={f.name:np.asarray([getattr(j,f.name) for j in jobs]) for f in dataclasses.fields(jobs[0]) if f.name!='meta'}
 a=fields['arrival_step']; y=fields['is_attack']; severity=fields['severity_true']; failure=fields['actual_backend_fail']; busy=fields['actual_busy_steps']
 starts=np.full(n,-1,dtype=np.int64); starts[ids]=steps
 ends=np.full(n,-1,dtype=np.int64); ends[ids]=steps+busy[ids]
 status=np.full(n,-1,dtype=np.int8)
 keys=['served_jobs','dropped_jobs_threshold','dropped_jobs_ttl','dropped_jobs_horizon']
 for k,key in enumerate(keys):
  sub=np.asarray(result[key],int); assert len(set(sub.tolist()))==len(sub) and np.all(status[sub]==-1); status[sub]=k
 assert np.all(status>=0)
 E=fields['pred_expected_consequence']; J=E/(1+fields['pred_busy_steps']/cfg.mean_pred_busy_steps+fields['pred_service_cost']/cfg.mean_pred_service_cost)
 admitted=np.ones(n,dtype=bool)
 if family in {'A','B'}:
  if spec['id']=='NONE_ACCEPT': admitted[:]=False
  elif spec['id']!='ALL_ACCEPT': admitted=(E if family=='A' else J)>=spec['value']
 assert np.array_equal(~admitted,status==1)
 assert np.all(starts[ids]>=a[ids]) and np.all(starts[ids]-a[ids]<=10)
 assert np.all(a[status==2]+11<T) and np.all(a[status==3]+11>=T)
 assert np.all(np.isfinite(fields['actual_service_time'])) and np.all(np.isfinite(fields['actual_service_cost']))
 assert np.all(busy==np.maximum(1,np.ceil(np.maximum(fields['actual_service_time'],0)/UNIT)).astype(int))
 trace=[]; audited=0
 for t,row in enumerate(result['step_trace']):
  active=int(np.sum((starts>=0)&(starts<t)&(ends>t))); free=cfg.slot_budget-active
  queue=np.flatnonzero(admitted&(a<=t)&(a+10>=t)&((starts<0)|(starts>=t)))
  if family in {'S','D'}:
   # Independent direct formula, same mathematical parent state; floating grouping is explicit.
   scores=cfg.v_weight*E[queue]/cfg.mean_pred_expected_consequence-cfg.clean_penalty*(1-fields['pred_attack_prob'][queue])-cfg.busy_penalty*(active/cfg.slot_budget+len(queue)/cfg.slot_budget)*(fields['pred_busy_steps'][queue]/cfg.mean_pred_busy_steps)+cfg.age_bonus*(t-a[queue])/10
   eligible=queue[scores>=cfg.admission_score_threshold]
   if family=='D': order=sorted(eligible.tolist(),key=lambda i:(a[i],i))
   else:
    scoremap=dict(zip(queue.tolist(),scores.tolist())); order=sorted(eligible.tolist(),key=lambda i:(-scoremap[i],a[i],i))
  else:
   eligible=queue
   order=sorted(queue.tolist(),key=(lambda i:(a[i],i)) if family=='A' else (lambda i:(-J[i],a[i],i)))
  actual=ids[steps==t].tolist(); assert actual==order[:free],(t,family,actual,order[:free])
  after=int(np.sum((starts>=0)&(starts<=t)&(ends>t)))
  assert after<=cfg.slot_budget and row['active_servers_after_action']==after
  assert row['selected_count']==len(actual) and row['available_servers_before_selection']==free
  assert row['queue_len_after_action']==len(queue)-len(actual) and row['arrivals_this_step']==int(np.sum(a==t))
  assert math.isclose(row['used_cost'],sum(jobs[i].actual_service_cost for i in actual),rel_tol=1e-12,abs_tol=1e-12)
  assert math.isclose(row['used_time'],sum(jobs[i].actual_service_time for i in actual),rel_tol=1e-12,abs_tol=1e-12)
  exp=np.flatnonzero((status==2)&(a+11==t))
  trace.append([t,len(queue),len(eligible),active,free,len(actual),len(exp),int(free>0 and len(queue)>0 and len(eligible)==0),int(after<cfg.slot_budget and len(queue)>len(actual)),int(len(queue)>free)])
  audited+=int(free>0 and len(queue)>0)
 trace=np.asarray(trace,dtype=np.int64)
 partitions={k:{'all':int(np.sum(status==i)),'attack':int(np.sum((status==i)&(y==1))),'clean':int(np.sum((status==i)&(y==0)))} for i,k in enumerate(['started','rejected','expired','horizon_unstarted'])}
 denominator=float(np.sum(severity[y==1])); numerator=float(np.sum(severity[(status==0)&(y==1)&(failure==0)])); attacks=int(np.sum(y==1)); cleans=int(np.sum(y==0))
 def ratio(num,den): return float(num/den) if den else None
 metrics=dict(R_weighted_nonfail=ratio(numerator,denominator),R_count=ratio(np.sum((status==0)&(y==1)),attacks),R_count_nonfail=ratio(np.sum((status==0)&(y==1)&(failure==0)),attacks),clean_starts=partitions['started']['clean'],clean_ratio=ratio(partitions['started']['clean'],cleans),starts=len(ids),attack_starts=partitions['started']['attack'],busy_steps=int(sum(busy[ids])),occupied_in_horizon=int(sum(np.minimum(ends[ids],T)-steps)),utilization=float(sum(np.minimum(ends[ids],T)-steps)/(T*cfg.slot_budget)),failures=int(sum(failure[ids])),attack_failures=int(sum(failure[ids[y[ids]==1]])),clean_failures=int(sum(failure[ids[y[ids]==0]])),recorded_cost=float(sum(fields['actual_service_cost'][ids])),recorded_time=float(sum(fields['actual_service_time'][ids])),cost_per_step=float(sum(fields['actual_service_cost'][ids])/T),D95=float(np.quantile(steps-a[ids],.95)) if len(ids) else None,precision=ratio(partitions['started']['attack'],len(ids)),denominator=denominator,numerator=numerator,expired=partitions['expired']['all'],horizon=partitions['horizon_unstarted']['all'],final_active_Tminus1=int(np.sum(ends>T-1)),unfinished_T=int(np.sum(ends>T)),idle_with_waiting=int(sum(trace[:,8])),competition_steps=int(sum(trace[:,9])),no_eligible_with_free_waiting=int(sum(trace[:,7])),expiry_with_free_slot=int(sum(trace[trace[:,4]>0,6])))
 for k in ['final_active_Tminus1','unfinished_T']:
  cutoff=T-1 if k=='final_active_Tminus1' else T
  metrics[k+'_attack']=int(np.sum((ends>cutoff)&(y==1))); metrics[k+'_clean']=int(np.sum((ends>cutoff)&(y==0)))
 summary=result['summary']
 for k,old in [('R_weighted_nonfail','weighted_attack_recall_no_backend_fail'),('R_count','attack_recall'),('clean_ratio','clean_service_ratio'),('recorded_cost','total_service_cost'),('recorded_time','total_service_time'),('D95','queue_delay_p95'),('utilization','server_utilization')]:
  if metrics[k] is not None: assert math.isclose(metrics[k],summary[old],rel_tol=1e-12,abs_tol=1e-12),(k,metrics[k],summary[old])
 assert metrics['failures']==summary['total_backend_fail']
 strata=[]
 for s in sorted(set(severity[y==1].tolist())):
  mask=(y==1)&(severity==s); strata.append(dict(severity_proxy=s,candidates=int(sum(mask)),starts=int(sum(mask&(status==0))),nonfail_starts=int(sum(mask&(status==0)&(failure==0)))))
 return dict(metrics=metrics,partitions=partitions,status=status,start_step=starts,busy_until=ends,start_ids=ids,start_steps=steps,diagnostic_trace=trace,decision_steps_checked=audited,strata=strata,undefined=[k for k,v in metrics.items() if v is None],fallback='MISSING_IN_HISTORICAL_SCHEMA',physical_success='NOT_INFERRED')
