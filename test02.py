import numpy as np

p = 'metric/case14/metric_event_trigger_mode_0_0.03_1.1.npy'
m = np.load(p, allow_pickle=True).item()

settings = list(m['recover_deviation'].keys())

def mean_safe(x):
    return float(np.mean(x)) if len(x) > 0 else None

def get_group(metric_name, s):
    v = m[metric_name]
    if isinstance(v, dict):
        return v.get(s, [])
    else:
        return None   # 不是按攻击设置分组的，就先不在每组里硬取

for s in settings:
    print(f'\n=== {s} ===')

    tp_trigger = len(m['recover_deviation'][s])
    print('DDD trigger count:', tp_trigger, '/ 200')
    print('DDD trigger rate :', tp_trigger / 200)

    print('pre_dev mean     :', mean_safe(get_group('pre_deviation', s)))
    print('recover_dev mean :', mean_safe(get_group('recover_deviation', s)))
    print('fail rate        :', mean_safe(get_group('fail', s)))
    print('stage1 eff mean  :', mean_safe(get_group('mtd_stage_one_eff', s)))
    print('stage2 eff mean  :', mean_safe(get_group('mtd_stage_two_eff', s)))
    print('stage1 hid mean  :', mean_safe(get_group('mtd_stage_one_hidden', s)))
    print('stage2 hid mean  :', mean_safe(get_group('mtd_stage_two_hidden', s)))
    print('obj1 mean        :', mean_safe(get_group('obj_one', s)))
    print('obj2 mean        :', mean_safe(get_group('obj_two', s)))

print('\n=== global metrics (not grouped by attack setting) ===')
for k in ['post_mtd_opf_converge', 'cost_no_mtd', 'cost_with_mtd_one', 'cost_with_mtd_two',
          'mtd_stage_one_time', 'mtd_stage_two_time', 'worst_primal', 'worst_dual',
          'varrho_summary', 'x_ratio_stage_one', 'x_ratio_stage_two']:
    v = m.get(k, None)
    if v is None:
        print(k, 'missing')
    elif isinstance(v, dict):
        print(k, 'is dict, grouped')
    else:
        try:
            print(k, 'len =', len(v), 'mean =', mean_safe(v))
        except:
            print(k, 'type =', type(v).__name__)