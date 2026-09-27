"""Exploratory position-preserving teacher affinity; not a trained module."""
import numpy as np
from probe_sam_group_compression import main, ROOT

def position_code(v,weights,kind):
    z=v[:,:,:64]
    if kind=='identity':
        return z.reshape(len(z),256)
    if kind=='uniform':
        affinity=np.full((len(z),4,4),.25)
    else:
        p=np.asarray(weights[kind],dtype=np.float64)
        affinity=p[:,:,None]*p[:,None,:]+(1-p[:,:,None])*(1-p[:,None,:])
        affinity/=np.maximum(affinity.sum(-1,keepdims=True),1e-6)
    mixed=.5*z+.5*np.einsum('nij,njc->nic',affinity,z)
    return mixed.reshape(len(z),256)

if __name__=='__main__':
    # No invented response on a constant signal; all codes have same shape.
    v=np.ones((2,4,128))
    weights={k:np.tile([0,0,1,1],(2,1)) for k in ('sam','box','sam_shift8')}
    for k in ('identity','uniform','sam','box','sam_shift8'):
        out=position_code(v,weights,k)
        assert out.shape==(2,256) and np.allclose(out,1)
    # Hard membership forbids cross-class mixing before residual combination.
    v=np.zeros((2,4,128)); v[:,2:]=1
    assert np.allclose(position_code(v,weights,'sam'),v[:,:,:64].reshape(2,256))
    assert not np.allclose(position_code(v,weights,'uniform'),v[:,:,:64].reshape(2,256))
    extras={f'position_{k}':lambda v,w,k=k:position_code(v,w,k) for k in ('identity','uniform','sam','box','sam_shift8')}
    main(extra_encoders=extras,output_dir=ROOT/'reports/117_sam_position_affinity')
