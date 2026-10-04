"""Small CPU tests for spatial axes, gradients and no-teacher inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.zoo.dfine.sam_side_search import SAMSideSearch
from src.zoo.dfine.sam_query_evidence_reader import SAMQueryEvidenceReader


torch.set_num_threads(2); torch.manual_seed(0)
model=SAMSideSearch(SAMQueryEvidenceReader(evidence_only=True)); model.topk=3
s4=torch.randn(2,64,16,20); s8=torch.randn(2,256,8,10)
boxes=torch.tensor([.5,.5,.25,.25]).repeat(2,5,1); logits=torch.randn(2,5,1)
result=model(s4,s8,logits,boxes,(64,80))
assert torch.equal(result['boxes'],boxes)
assert result['selected']['scores'].shape==(2,3,4,17)
# A soft target at the exact box is the central bin on every side.
score=torch.zeros(1,4,17,requires_grad=True)
ce,valid=model.position_loss(score,boxes[0,:1],boxes[0,:1]); ce.backward()
assert valid.all() and (score.grad[0,:,8]<0).all()
assert (score.grad[0,:,:8]>0).all()
# Geometry sanity: all possible independent movements preserve positive width.
model.score[-1].weight.data.normal_(std=.05)
result=model(s4,s8,logits,boxes,(64,80)); assert (result['boxes'][...,2:]>0).all()
mask=torch.zeros(1,64,80); mask[:,24:40,30:50]=1
targets=[{'boxes':boxes[0,:1],'masks':mask,'sam_quality':torch.tensor(1.)},{'boxes':torch.empty(0,4)}]
matches=[(result['indices'][0,:1],torch.tensor([0])),(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))]
loss,stats=model.loss(result,boxes,targets,matches,1.); assert torch.isfinite(loss)
loss.backward(); assert model.shape_head[-1].weight.grad.abs().sum()>0
model.eval()
with torch.no_grad(): evaluation=model(s4,s8,logits,boxes,(64,80))
assert 'memory' not in evaluation and 'shape_logits' not in evaluation
# No mask/no GT still permits valid inference, and empty-target loss is finite.
model.train(); result=model(s4,s8,logits,boxes,(64,80))
empty=[{'boxes':torch.empty(0,4)} for _ in range(2)]
pairs=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(2)]
loss,_=model.loss(result,boxes,empty,pairs); assert torch.isfinite(loss)
print('PASS: candidate axes, zero identity, central-bin supervision, valid boxes, shape gradients, empty GT and teacher-free inference')
