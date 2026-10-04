"""CPU synthetic contracts, independent of datasets/checkpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.zoo.dfine.sam_boundary_refiner import SAMBoundaryRefiner,boundary_teacher,refinement_loss


def main():
    torch.manual_seed(5); torch.set_num_threads(2)
    for hw in [(64,80),(96,128)]:
        model=SAMBoundaryRefiner(topk=4,roi_size=16)
        memory=torch.randn(2,64,hw[0]//4,hw[1]//4)
        queries=torch.randn(2,7,128); logits=torch.randn(2,7,1)
        boxes=torch.tensor([.5,.5,.25,.25])[None,None].repeat(2,7,1)
        boxes[:,1]=torch.tensor([.03,.03,.04,.04])
        boxes[:,2]=torch.tensor([.99,.99,.0001,.0001])
        output=model(memory,queries,logits,boxes,hw)
        assert torch.equal(output['boxes'],boxes)
        assert output['attention'].shape==(2,4,4,256)
        assert torch.allclose(output['attention'].sum(-1),torch.ones(2,4,4),atol=1e-5)
        # Explicitly change head, test valid boxes and no cross-image/candidate leakage.
        torch.nn.init.normal_(model.head[-1].weight,std=.1)
        output=model(memory,queries,logits,boxes,hw)
        chosen=torch.zeros(2,7,dtype=torch.bool).scatter(1,output['indices'],True)
        assert torch.equal(output['boxes'][~chosen],boxes[~chosen])
        assert (output['boxes'][...,2:]>0).all()
        empty=[{'boxes':torch.empty(0,4)} for _ in range(2)]
        pairs=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(2)]
        loss,_=refinement_loss(output,boxes,empty,pairs)
        assert torch.isfinite(loss); loss.backward()
        # Missing masks: regression still trains, and teacher is not a forward argument.
        targets=[{'boxes':torch.tensor([[.5,.5,.20,.20]])} for _ in range(2)]
        pairs=[(output['indices'][i,:1],torch.tensor([0])) for i in range(2)]
        output=model(memory,queries,logits,boxes,hw)
        loss,stats=refinement_loss(output,boxes,targets,pairs)
        assert torch.isfinite(loss) and stats['teacher_valid']==0
    model=SAMBoundaryRefiner(topk=1,roi_size=32)
    logits=torch.zeros(1,1,1); boxes=torch.tensor([[[.5,.5,.4,.4]]])
    output=model(torch.randn(1,64,16,16),torch.randn(1,1,128),logits,boxes,(64,64))
    mask=torch.zeros(64,64); mask[23:41,23:41]=1
    teacher,ok=boundary_teacher(mask,output['grid'][0,0]); assert ok
    assert torch.allclose(teacher.sum(-1),torch.ones(4),atol=1e-6)
    targets=[{'boxes':boxes[0], 'masks':mask[None], 'sam_quality':torch.tensor(1.)}]
    loss,stats=refinement_loss(output,boxes,targets,[(torch.tensor([0]),torch.tensor([0]))])
    loss.backward()
    assert stats['teacher_valid']==1
    assert model.roles.grad.abs().sum()>0
    # Zero head + SAM shape gradient must reach positional and query selection machinery.
    assert model.position.weight.grad.abs().sum()>0
    print('PASS: zero identity, finite valid boxes, batch/candidate axes, dynamic size, empty GT, missing teacher, shape gradients')


if __name__=='__main__': main()
