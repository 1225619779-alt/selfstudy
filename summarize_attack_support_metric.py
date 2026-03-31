import argparse
import numpy as np
from pathlib import Path

GROUPS = ['(1,0.2)','(1,0.3)','(2,0.2)','(2,0.3)','(3,0.2)','(3,0.3)']

def load_npy(path):
    return np.load(path, allow_pickle=True).item()

def mean_safe(x):
    arr = np.asarray(list(x), dtype=float)
    return float(arr.mean()) if arr.size else float('nan')

def count_true(x):
    return int(np.asarray(list(x), dtype=bool).sum())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metric', default='metric/case14/metric_event_trigger_tau_0.03_mode_0_0.03_1.1.npy')
    args = p.parse_args()
    path = Path(args.metric)
    if not path.exists():
        raise FileNotFoundError(f'Not found: {path}')
    d = load_npy(path)
    print('==== Attack supporting summary ====')
    print('metric_file:', path)
    total_alarm = 0
    total_trigger = 0
    for g in GROUPS:
        tp = d['TP_DDD'].get(g, [])
        alarm_n = count_true(tp)
        trig_n = sum(bool(v) for v in d.get('trigger_after_verification', {}).get(g, []))
        skip_n = sum(bool(v) for v in d.get('skip_by_verification', {}).get(g, []))
        total_alarm += alarm_n
        total_trigger += trig_n
        print(f'{g}: DDD_alarm={alarm_n} trigger_after_gate={trig_n} skip_by_gate={skip_n}')
    print('total_DDD_alarm:', total_alarm)
    print('total_trigger_after_gate:', total_trigger)
    if total_alarm:
        print('overall_trigger_rate:', round(total_trigger/total_alarm, 6))
    print('\n---- Per-alarm burdens on triggered attacks ----')
    fail_vals = []
    if 'fail' in d:
        for g in GROUPS:
            fail_vals.extend(list(d['fail'].get(g, [])))
    if fail_vals:
        print('fail_per_triggered_alarm:', mean_safe(fail_vals))
    if 'mtd_stage_one_time' in d:
        print('mean_stage_one_time_triggered:', mean_safe(d['mtd_stage_one_time']))
    if 'mtd_stage_two_time' in d:
        print('mean_stage_two_time_triggered:', mean_safe(d['mtd_stage_two_time']))
    if 'cost_no_mtd' in d and 'cost_with_mtd_one' in d:
        dc1 = []
        for g in GROUPS:
            a = np.asarray(d['cost_no_mtd'].get(g, []), dtype=float)
            b = np.asarray(d['cost_with_mtd_one'].get(g, []), dtype=float)
            if a.size and a.size == b.size:
                dc1.extend((b-a).tolist())
        if dc1:
            print('mean_delta_cost_stage_one_triggered:', mean_safe(dc1))
    if 'cost_no_mtd' in d and 'cost_with_mtd_two' in d:
        dc2 = []
        for g in GROUPS:
            a = np.asarray(d['cost_no_mtd'].get(g, []), dtype=float)
            b = np.asarray(d['cost_with_mtd_two'].get(g, []), dtype=float)
            if a.size and a.size == b.size:
                dc2.extend((b-a).tolist())
        if dc2:
            print('mean_delta_cost_stage_two_triggered:', mean_safe(dc2))

if __name__ == '__main__':
    main()
