"""Teacher validity must not remove GT or turn unknown targets into background."""
import json

import numpy as np
import torch
from PIL import Image

from src.data.dataset.coco_dataset import CocoDetection
from src.data.transforms._transforms import SanitizeBoundingBoxes
from src.data._misc import convert_to_tv_tensor
from src.zoo.dfine.sam_group_contrast import selection, group_loss, raster
from src.zoo.dfine.sam_query_evidence_reader import sqer_shape_loss, _region_distributions


def fixture():
    size = 128
    boxes = torch.tensor([[.30,.50,.20,.20], [.48,.50,.15,.20]])
    masks = torch.stack([raster(box[None],size,size) for box in boxes]).to(torch.uint8)
    target = dict(boxes=boxes, masks=masks, sam_quality=torch.tensor([1.]))
    axes = []
    for cx,cy,w,h in boxes:
        x = torch.linspace(cx-w, cx+w, 16)*2-1
        y = torch.linspace(cy-h, cy+h, 16)*2-1
        yy,xx = torch.meshgrid(y,x,indexing='ij')
        axes.append(torch.stack([xx,yy],-1))
    attention = torch.full((1,2,3,256),1/256,requires_grad=True)
    outputs = dict(sqer_attention=attention,sqer_roi_grid=torch.stack(axes)[None],sqer_query_indices=torch.tensor([[3,7]]))
    matches = [(torch.tensor([3,7]),torch.tensor([0,1]))]
    return target,outputs,matches


def test_partial_teacher_preserves_valid_query_only():
    target,outputs,matches = fixture()
    target['sam_instance_valid'] = torch.tensor([True,False])
    loss,diagnostics = sqer_shape_loss(outputs,[target],matches)
    assert torch.isfinite(loss) and diagnostics['sqer_supervised_queries'].item()==1
    loss.backward()
    assert outputs['sqer_attention'].grad[0,0].abs().sum()>0
    assert outputs['sqer_attention'].grad[0,1].abs().sum()==0
    reference = loss.detach()
    target['masks'][1] = 1
    other,_ = sqer_shape_loss(outputs,[target],matches)
    torch.testing.assert_close(other,reference,rtol=0,atol=0)


def test_sgc_never_labels_rejected_object_as_background():
    for source in ('sam','box'):
        target,_,_ = fixture()
        target['sam_instance_valid']=torch.tensor([True,False])
        p,n=selection(target,(32,32),source)
        rejected = torch.nn.functional.interpolate(raster(target['boxes'][1:],128,128)[None,None],size=(32,32),mode='area')[0,0]>0
        assert not rejected.flatten()[p].any()
        assert not rejected.flatten()[n].any()
        reference = (p.clone(),n.clone())
        target['masks'][1]=1
        got=selection(target,(32,32),source)
        assert all(torch.equal(a,b) for a,b in zip(reference,got))


def test_all_invalid_has_finite_zero_auxiliary_losses():
    target,outputs,matches = fixture()
    target['sam_instance_valid']=torch.tensor([False,False])
    value,diag=sqer_shape_loss(outputs,[target],matches)
    assert value.item()==0 and diag['sqer_supervised_queries'].item()==0
    feature=torch.randn(1,16,32,32,requires_grad=True)
    value=value+group_loss(feature,[target],'sam')
    value.backward()
    assert feature.grad.abs().sum()==0


def test_all_valid_is_exactly_legacy_loss():
    target,outputs,matches=fixture()
    legacy,_=sqer_shape_loss(outputs,[target],matches)
    features=torch.randn(1,16,32,32)
    legacy_group=group_loss(features,[target],'sam')
    target['sam_instance_valid']=torch.tensor([True,True])
    current,_=sqer_shape_loss(outputs,[target],matches)
    assert torch.equal(legacy,current)
    assert torch.equal(legacy_group,group_loss(features,[target],'sam'))


def test_unknown_pixels_receive_no_region_teacher_mass():
    mask=torch.zeros(16,16);mask[5:11,5:11]=1
    ignored=torch.zeros(16,16,dtype=torch.bool);ignored[:,10:]=True
    distribution,valid,_=_region_distributions(mask,ignored=ignored)
    assert valid.item()
    assert distribution.reshape(3,16,16)[:,:,10:].sum()==0
    torch.testing.assert_close(distribution.sum(-1),torch.ones(1,3))


def test_dataset_retains_all_boxes_and_loads_only_valid_masks(tmp_path):
    images=tmp_path/'images';images.mkdir()
    Image.new('RGB',(64,64)).save(images/'a.png')
    annotation=tmp_path/'annotations.json'
    annotation.write_text(json.dumps(dict(images=[dict(id=1,file_name='a.png',width=64,height=64)],
        categories=[dict(id=0,name='drone')],annotations=[
            dict(id=i+1,image_id=1,category_id=0,bbox=box,area=box[2]*box[3],iscrowd=0)
            for i,box in enumerate([[8,8,16,16],[40,40,12,12]])])),encoding='utf-8')
    cache=tmp_path/'cache';(cache/'masks').mkdir(parents=True)
    mask=np.zeros((64,64),dtype=np.uint8);mask[8:24,8:24]=255
    Image.fromarray(mask).save(cache/'masks/good.png')
    (cache/'records.json').write_text(json.dumps([dict(image_id=1,accepted=True,supervision_weight=1.,
        instance_mask_files=['good.png','unsaved-invalid.png'],instance_mask_valid=[True,False])]),encoding='utf-8')
    dataset=CocoDetection(str(images),str(annotation),transforms=None,sam_mask_root=str(cache))
    _,target=dataset.load_item(0)
    assert len(target['boxes'])==2 and target['masks'].shape==(2,64,64)
    assert target['sam_instance_valid'].tolist()==[True,False]
    assert target['masks'][1].sum()==0


def test_sanitize_keeps_teacher_validity_aligned_with_retained_boxes():
    mask=torch.zeros(3,64,64,dtype=torch.uint8);mask[0,8:24,8:24]=1;mask[2,32:48,32:48]=1
    boxes=convert_to_tv_tensor(torch.tensor([[8.,8.,24.,24.],[30.,30.,30.,40.],[32.,32.,48.,48.]]),key='boxes',spatial_size=(64,64))
    target=dict(boxes=boxes,labels=torch.zeros(3,dtype=torch.long),masks=convert_to_tv_tensor(mask,key='masks'),
                sam_instance_valid=torch.tensor([True,False,True]))
    _,result=SanitizeBoundingBoxes()(Image.new('RGB',(64,64)),target)
    assert len(result['boxes'])==2 and result['sam_instance_valid'].tolist()==[True,True]
    assert result['sam_instance_valid'].dtype==torch.bool
