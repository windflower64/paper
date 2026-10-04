"""Run the focused regression tests without installing a test framework."""
import importlib.util
import inspect
import json
import sys
import tempfile
from pathlib import Path
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def main():
    torch.set_num_threads(4)
    rows=[]
    for filename in ('test_multidata_multi_instance_sam.py','test_sam_instance_validity.py'):
        spec=importlib.util.spec_from_file_location(filename.removesuffix('.py'),REPO/'tests'/filename)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for name,value in vars(module).items():
            if not name.startswith('test_') or not callable(value):continue
            if inspect.signature(value).parameters:
                with tempfile.TemporaryDirectory(prefix='sam-instance-tests-') as temporary:
                    value(Path(temporary))
            else:
                value()
            rows.append({'test':name,'status':'PASS'})
            print(name+' PASS',flush=True)
    destination=REPO.parent/'reports/TRI_MODAL_RGBT_811_REPAIR_V1/unit_tests.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({'status':'PASS','checks':rows},indent=2),encoding='utf-8')


if __name__=='__main__':main()
