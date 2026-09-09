"""Pure unit/contract checks; no queue trajectories or scientific production code."""
import copy,dataclasses,json,unittest,types
import numpy as np
import reproduce as r
class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conf=json.loads((r.R/'data/CONFIG.json').read_bytes())
        cls.bank=r.load_numeric(r.R/'data/BANK.npz')
        cls.tag=cls.conf['tags'][0]
        cls.j=cls.bank[cls.tag]['jobs_evaluator_only']
    def test_all_saved_inputs_and_configs(self):
        for tag,b in self.bank.items():
            r.validate_jobs(b['jobs_evaluator_only'],540)
            for B in (1,2):r.validate_config(self.conf['configs'][tag][str(B)])
    def test_hashes(self):
        self.assertEqual(r.sha256_file(r.R/'data/BANK.npz'),self.conf['bank_sha256'])
        self.assertEqual(r.sha256_file(r.R/'data/HISTORICAL_REFERENCE.npz'),self.conf['gold_sha256'])
    def test_truth_and_completion_not_policy_visible(self):
        job=r.engine.AlarmJob(**{k:v[0].item() for k,v in self.j.items()})
        visible=r.project(job,job.arrival_step)
        self.assertEqual(set(vars(visible)),set(r.VISIBLE))
        for k in ('is_attack','severity_true','actual_service_time','actual_backend_fail','actual_busy_steps','completion_time'):
            self.assertFalse(hasattr(visible,k))
        changed=copy.copy(job);changed.is_attack=1-job.is_attack;changed.actual_busy_steps=999
        self.assertEqual(vars(visible),vars(r.project(changed,job.arrival_step)))
    def test_future_rejected(self):
        job=r.engine.AlarmJob(**{k:v[0].item() for k,v in self.j.items()})
        with self.assertRaises(ValueError):r.project(job,job.arrival_step-1)
    def test_missing_outcome_not_defaulted(self):
        j=dict(self.j);j.pop('actual_backend_fail')
        with self.assertRaises(ValueError):r.validate_jobs(j,540)
    def test_missing_duration_not_defaulted(self):
        j={k:v.copy() for k,v in self.j.items()};j['actual_busy_steps'][0]=0
        with self.assertRaises(ValueError):r.validate_jobs(j,540)
    def test_truth_cost_budget_prohibited(self):
        c=dict(self.conf['configs'][self.tag]['1']);c['window_cost_budget']=10;c['cost_budget_window_steps']=10
        with self.assertRaises(ValueError):r.validate_config(c)
    def test_no_negative_or_nonbinary_labels(self):
        j={k:v.copy() for k,v in self.j.items()};j['is_attack'][0]=-1
        with self.assertRaises(ValueError):r.validate_jobs(j,540)
    def test_planner_rejects_extra_truth(self):
        from p05b_scheduler_core import observation_from_visible
        with self.assertRaises((ValueError,TypeError)):
            observation_from_visible(now=0,capacity=1,waiting=[dict(job_id=0,arrival=0,latest_start=10,predicted_value=1.,predicted_duration=1.,is_attack=1)],active=[])
    def test_param_grid_is_bounded(self):
        n=sum((4+(B==1))*2 for tag in self.conf['tags'] for B in (1,2))
        self.assertEqual(n,144)
if __name__=='__main__':unittest.main(verbosity=2)
