"""Same S16 SBR3 in both arms; vary SGC scales only."""
import argparse
from pathlib import Path
import torch
import train_sbr3_context as context
from preflight_sqer1 import seed_all,sha256_file

joint=context.joint
RUN=joint.task.engine.ROOT/'outputs/SCALE_COMBINATION_20261005'


def setup(batch,arm):
    cfg,source,refiner,criterion=context.setup(batch,'CONTEXT')
    source.model.sgc_multiscale=arm=='MULTISCALE'
    return cfg,source,refiner,criterion


def evaluate(cfg,source,refiner):
    normal=context.evaluate(cfg,source,refiner)
    hook=refiner.context_score.register_forward_hook(lambda m,a,o:torch.zeros_like(o))
    try:
        disabled=context.evaluate(cfg,source,refiner)
    finally:
        hook.remove()
    normal['s16_disabled_diagnostic']=disabled
    return normal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--arm',choices=['S8','MULTISCALE']);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=16);args=p.parse_args()
    joint.RUN=RUN;joint.setup=setup
    joint.task.readout=context.readout;joint.task.evaluate=evaluate
    seed_all(0);torch.set_num_threads(4)
    if args.preflight:joint.preflight(args)
    elif args.arm:
        joint.train(args,extra_protocol={'module':'SCALE_COMBINATION', 'sbr_scales':[4,8,16],
            'sgc_scales':[4,8,16] if args.arm=='MULTISCALE' else [8],
            'm_variant':'original S8/P16 inputs; P16 write; frozen parameters',
            'selection_metric':'normal S16-enabled AP; disabled scores recorded as diagnostic only',
            'initialization_note':'shared historical baseline and SBR3-SAM head, zero new context score; not resumed from a scale experiment best',
            'experiment_hashes':{str(f):sha256_file(f) for f in [Path(__file__),Path(context.__file__),
                joint.task.engine.REPO/'src/zoo/dfine/sam_side_search_multiscale.py']}})
    else:p.error('Specify preflight or arm')
