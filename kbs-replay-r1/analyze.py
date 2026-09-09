"""Analyze saved run outputs only. No queue, planner, prediction or physical calls."""
from pathlib import Path
import argparse,csv,json,itertools
import numpy as np
from safe_numeric_io import load_numeric,write_json_new
R=Path(__file__).resolve().parent
def readcsv(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def csvout(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    run=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    b=readcsv(run/'BASELINE_METRICS.csv');s=readcsv(run/'SENSITIVITY_ALL.csv');bank=load_numeric(R/'data/BANK.npz')
    # All intersections are at bottom-level source endpoint, not attack-event ID.
    matrix=[];sets={t:set(x['full_endpoints_evaluator_only']['idx_summary'].tolist()) for t,x in bank.items()}
    for t,u in itertools.combinations(sets,2):
        shared=sets[t]&sets[u];f=bank[t]['full_endpoints_evaluator_only'];g=bank[u]['full_endpoints_evaluator_only']
        fm={int(v):i for i,v in enumerate(f['idx_summary'])};gm={int(v):i for i,v in enumerate(g['idx_summary'])}
        differing=sum(f['scenario_label'][fm[k]]!=g['scenario_label'][gm[k]] for k in shared)
        matrix.append(dict(left=t,right=u,left_size=len(sets[t]),right_size=len(sets[u]),shared_source_endpoints=len(shared),overlap_fraction_of_each_540=len(shared)/540,jaccard=len(shared)/len(sets[t]|sets[u]),different_assigned_scenario_at_shared_source=differing))
    csvout(out/'SOURCE_OVERLAP_PAIRS.csv',matrix)
    union=set.union(*sets.values());counts={k:sum(k in v for v in sets.values()) for k in union}
    csvout(out/'SOURCE_ENDPOINT_MULTIPLICITY.csv',[dict(source_endpoint=k,number_of_replays=counts[k]) for k in sorted(counts)])
    overlap=dict(total_occurrences=sum(map(len,sets.values())),distinct_source_endpoints=len(union),min_source=min(union),max_source=max(union),adjacent_shared=480,adjacent_overlap_fraction=480/540,pair_overlap_min=min(x['shared_source_endpoints'] for x in matrix),pair_overlap_max=max(x['shared_source_endpoints'] for x in matrix),disjoint_pairs=sum(x['shared_source_endpoints']==0 for x in matrix),average_multiplicity=float(np.mean(list(counts.values()))),maximum_multiplicity=max(counts.values()),strict_independent_split_from_existing_eight_full_replays=False,note='Source endpoint is not an independent attack-event ID. Different scenario labels on shared endpoints are retained, not arbitrarily deduplicated. No estimated effective sample size or inferential p-value.')
    write_json_new(out/'SOURCE_OVERLAP_SUMMARY.json',overlap)
    # Group means are equal-weight record means, no pseudo-independent intervals.
    summary=[]
    for B,param,factor in sorted(set((int(x['B']),x['parameter'],float(x['factor'])) for x in s)):
        sub=[x for x in s if int(x['B'])==B and x['parameter']==param and float(x['factor'])==factor]
        v=lambda k:np.array([float(x[k]) for x in sub])
        summary.append(dict(B=B,parameter=param,factor=factor,replays=len(sub),mean_delta_Rw_pp=float(v('delta_Rw_pp').mean()),min_delta_Rw_pp=float(v('delta_Rw_pp').min()),max_delta_Rw_pp=float(v('delta_Rw_pp').max()),mean_delta_Rn_pp=float(v('delta_Rn_pp').mean()),mean_delta_busy=float(v('delta_busy').mean()),mean_delta_starts=float(v('delta_starts').mean()),min_selected_jaccard=float(v('selected_jaccard').min()),mean_selected_jaccard=float(v('selected_jaccard').mean()),changed_served_sets=int(np.sum(v('selected_symmetric_difference')>0))))
    csvout(out/'SENSITIVITY_SUMMARY.csv',summary)
    resource=[];kmap={(x['tag'],int(x['B']),x['method']):x for x in b}
    for B in (1,2):
        for tag in bank:
            for left,right in [('ROLLING_CURRENT_QUEUE_NONDELAY','ATC_START'),('A_Q25_FIFO','S0'),('ATC_START','S0'),('ROLLING_CURRENT_QUEUE_NONDELAY','S0')]:
                a=kmap[tag,B,left];c=kmap[tag,B,right]
                d=lambda k:float(a[k])-float(c[k])
                resource.append(dict(tag=tag,B=B,left=left,right=right,delta_Rw_pp=100*d('R_weighted_nonfail'),delta_Rn_pp=100*d('R_count_nonfail'),delta_busy=d('busy_steps'),delta_clean=d('clean_starts'),delta_starts=d('starts'),delta_cost=d('recorded_cost'),delta_D95=d('D95'),equal_busy_and_clean=d('busy_steps')==0 and d('clean_starts')==0,within_right_busy_clean_caps=d('busy_steps')<=0 and d('clean_starts')<=0))
    csvout(out/'RESOURCE_PAIRED_ALL.csv',resource)
    groups=[]
    for B,left,right in sorted(set((x['B'],x['left'],x['right']) for x in resource)):
        sub=[x for x in resource if (x['B'],x['left'],x['right'])==(B,left,right)]
        same=[x for x in sub if x['equal_busy_and_clean']];cap=[x for x in sub if x['within_right_busy_clean_caps']]
        groups.append(dict(B=B,left=left,right=right,all_n=len(sub),mean_delta_Rw_pp=float(np.mean([x['delta_Rw_pp'] for x in sub])),positive_Rw=sum(x['delta_Rw_pp']>1e-10 for x in sub),negative_Rw=sum(x['delta_Rw_pp']<-1e-10 for x in sub),min_delta_busy=min(x['delta_busy'] for x in sub),max_delta_busy=max(x['delta_busy'] for x in sub),equal_busy_clean_n=len(same),equal_subset_mean_delta_Rw_pp=float(np.mean([x['delta_Rw_pp'] for x in same])) if same else None,within_caps_n=len(cap),within_caps_mean_delta_Rw_pp=float(np.mean([x['delta_Rw_pp'] for x in cap])) if cap else None))
    csvout(out/'RESOURCE_PAIR_SUMMARY.csv',groups)
    # Existing fixed P02 menu at each S0 budget: disclose finite retrospective cap
    # support, never pass realized cap values back into an online policy.
    menu=json.loads((R/'data/EXISTING_MENU_METRICS.json').read_bytes());feas=[]
    for B in (1,2):
        for tag in bank:
            ref=kmap[tag,B,'S0'];limitL=float(ref['busy_steps']);limitN=float(ref['clean_starts'])
            for family in ('A','B'):
                subset=[x for x in menu if x['tag']==tag and int(x['B'])==B and x['family']==family]
                for x in subset:
                    m=x['metrics'];feas.append(dict(tag=tag,B=B,family=family,setting=x['menu']['id'],busy_cap=limitL,clean_cap=limitN,feasible=m['busy_steps']<=limitL and m['clean_starts']<=limitN,busy=m['busy_steps'],clean=m['clean_starts'],Rw=m['R_weighted_nonfail'],delta_Rw_pp=100*(m['R_weighted_nonfail']-float(ref['R_weighted_nonfail']))))
    csvout(out/'EXISTING_MENU_CAP_FEASIBILITY.csv',feas)
    write_json_new(out/'ANALYSIS_SUMMARY.json',dict(overlap=overlap,sensitivity=summary,resources=groups,interpretation='Retrospective, no parameter selected, no independence or exact resource-matching claim from subset means. Subsets are outcome-conditioned descriptive diagnostics.',new_scientific_calls=0))
    print(json.dumps(dict(overlap=overlap,sensitivity=summary,resources=groups),indent=2))
if __name__=='__main__':main()
