import torch
import torch.distributed as d
d.init_process_group("gloo")
x=torch.tensor(float(d.get_rank()+1))
d.all_reduce(x)
assert x.item()==3.0
print("rank",d.get_rank(),"sum",x.item(),flush=True)
d.destroy_process_group()
