"""Only frozen bin lookup and scalar fusion; excludes detector and fitting."""
import json,numpy as np
from fusion_copy import _fuse_posteriors
class FrozenScores:
    def __init__(self,path,*,fusion_verify_weight,busy_time_unit,normalizers):
        z=np.load(path,allow_pickle=False)
        def dec(x):
            if isinstance(x,dict) and 'array_key' in x:return z[x['array_key']].copy()
            if isinstance(x,dict):return {k:dec(v) for k,v in x.items()}
            return x
        self.models=dec(json.loads(z['metadata_utf8'].tobytes()))
        self.weight=fusion_verify_weight;self.unit=busy_time_unit;self.normalizers=normalizers
    def lookup(self,m,x):
        a=np.asarray(x,dtype=float);out=np.full(a.shape,m['default_value'],dtype=float);f=np.isfinite(a)
        ix=np.clip(np.digitize(a[f],m['edges'][1:-1],right=False),0,len(m['values'])-1)
        out[f]=m['values'][ix];return out
    def predict(self,verify,ddd):
        m=self.models;v=np.asarray(verify);d=np.asarray(ddd)
        p=_fuse_posteriors(self.lookup(m['posterior_verify'],v),self.lookup(m['posterior_ddd'],d),verify_weight=self.weight)
        sev=self.weight*self.lookup(m['severity']['verify_score'],v)+(1-self.weight)*self.lookup(m['severity']['ddd_loss_recons'],d)
        fail=self.lookup(m['service']['backend_fail'],v);t=self.lookup(m['service']['service_time'],v)
        return dict(probability=p,severity=sev,failure=fail,service_time=t,cost=self.lookup(m['service']['service_cost'],v),value=p*np.maximum(sev,0)*(1-np.clip(fail,0,1)),duration=np.maximum(1,np.ceil(np.maximum(t,0)/self.unit)).astype(int))
