#!/usr/bin/env python3
import json, math, sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
bins=[("lt8",0,8),("8to16",8,16),("16to32",16,32),("32to48",32,48),("ge48",48,float("inf"))]
x=[math.sqrt(a["bbox"][2]*a["bbox"][3]) for a in d["annotations"] if not a.get("iscrowd",0)]
print(json.dumps({n:sum(lo<=v<hi for v in x) for n,lo,hi in bins}))
