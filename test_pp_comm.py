import datetime
import logging
import sys

global GLOBAL_START_TIME
GLOBAL_START_TIME = datetime.datetime.now()
import torch
import torch.nn.parallel
import os
import socket
import torch.distributed as dist
import argparse
from torch.profiler import profile, record_function, ProfilerActivity


def trace_func(func):
   def wrapper(*args, **kwargs):
      try:
         function_name = func.__func__.__qualname__
      except:
         function_name = func.__qualname__
      with record_function(function_name):
         return func(*args, **kwargs)
   return wrapper



class Model:
   def __init__(self, num_layers = 1, pp = 1, rank = 0, world_size = 1, device = None, logger=None):
      self.num_layers = num_layers
      self.pp = pp
      self.ppn = world_size // self.pp
      self.my_pp_rank = rank//self.ppn
      self.my_pp_group = rank%self.ppn
      self.world_size = world_size
      self.rank = rank
      self.rank_right = (rank + self.ppn)%world_size
      self.rank_left = (rank - self.ppn + world_size)%world_size
      self.device = device
      self.logger = logger
      self.tensor = torch.zeros(self.num_layers*1024).to(device, non_blocking=True)
      assert(self.tensor[0]==0)
      t0 = time.time()
      dist.broadcast(self.tensor, src=0)
      if self.rank==0:
         self.logger.info(f"broadcast in model.init: {time.time()-t0:.8f}")
   def reset(self):
      self.tensor = torch.zeros(self.num_layers*1024).to(self.device, non_blocking=True)
      
   @trace_func
   def send_to_right(self):
      if self.my_pp_rank != self.pp - 1:
         dist.send(tensor=self.tensor, dst=self.rank_right)
         if self.my_pp_group==0:
            self.logger.debug(f" {self.my_pp_rank} send_to_right {self.tensor[0]}")

   @trace_func
   def recv_from_left(self):
      if self.my_pp_rank != 0:
         dist.recv(tensor=self.tensor, src=self.rank_left)
         if self.my_pp_group==0:
            self.logger.debug(f" {self.my_pp_rank} recv_from_left {self.tensor[0]}")

   @trace_func
   def send_to_left(self):
      if self.my_pp_rank != 0:   
         dist.send(tensor=self.tensor, dst=self.rank_left)
         if self.my_pp_group==0:
            self.logger.debug(f" {self.my_pp_rank} send_to_left {self.tensor[0]}")

   @trace_func
   def recv_from_right(self):
      if self.my_pp_rank != self.pp - 1:
         dist.recv(tensor=self.tensor, src=self.rank_right)
         if self.my_pp_group==0:
            self.logger.debug(f" {self.my_pp_rank} recv_from_right {self.tensor[0]}")

   @trace_func
   def forward_comp(self):
      self.tensor += 1

   @trace_func
   def backward_comp(self):
      self.tensor += 2

   @trace_func
   def backward(self):
      self.recv_from_right()
      self.backward_comp()
      self.send_to_left()
      
   @trace_func
   def forward(self):
      self.recv_from_left()
      self.forward_comp()      
      self.send_to_right()
      
   @trace_func
   def comp_init(self):
      x = torch.ones(1024).to(self.device, non_blocking=True)
      x += 0
      x *= 0
      
   @trace_func   
   def comm_init(self):
      dist.barrier()
      x = torch.ones(1024).to(self.device, non_blocking=True)
      t0 = time.time()
      dist.broadcast(x, src=0)
      if self.rank==0:
         self.logger.info(f"broadcast in comm_init: {time.time()-t0:.8f}")
      t0 = time.time()
      if (self.my_pp_rank)%2==1:
         self.logger.debug(f"L{self.my_pp_rank}-r{self.my_pp_group} recv issued at {time.time()-t0:.8f}")      
         self.recv_from_left()
         self.logger.debug(f"L{self.my_pp_rank}-r{self.my_pp_group} received at {time.time()-t0:.8f}")
      else:
         self.logger.debug(f"L{self.my_pp_rank}-r{self.my_pp_group} send issued at {time.time()-t0:.8f}")            
         self.send_to_right()
         self.logger.debug(f"L{self.my_pp_rank}-r{self.my_pp_group} sent at {time.time()-t0:.8f}")      
      dist.barrier()
      t1 = time.time()
      if self.rank==0:
         self.logger.info(f"F: 0->1, 2->3, 4-> ...: {t1-t0:.8f}")

      t0 = time.time()
      if (self.my_pp_rank)%2==0:
         self.recv_from_right()
      else:
         self.send_to_left()
      if self.rank==0:
         self.logger.info(f"B: 0<-1, 2<-3, 4<- ...: {time.time() - t0:.8f}")
      dist.barrier()
         
      t0 = time.time()
      if (self.my_pp_rank)%2==0:
         self.recv_from_left()
      else:
         self.send_to_right()
      dist.barrier()
      t1 = time.time()      
      if self.rank==0:
         self.logger.info(f"F: 0, 1->2, 3->4,  ...: {t1-t0:.8f}")      

      t0 = time.time()
      if (self.my_pp_rank)%2==1:
         self.recv_from_right()
      else:
         self.send_to_left()
      dist.barrier()      
      if self.rank==0:
         self.logger.info(f"B: 0, 1<-2, 3<-4,  ...: {time.time() - t0:.8f}")
      dist.barrier()   
import time

def set_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--pp", default=1, type=int)
   parser.add_argument("--tp", default=1, type=int)
   parser.add_argument("--backend", default='ccl', type=str)
   parser.add_argument("--init-comm", action='store_true')
   parser.add_argument("--init-comp", action='store_true')
   parser.add_argument("--niters", default=10, type=int)
   parser.add_argument("--debug", action='store_true')
   parser.add_argument('--output', default="output.log", type=str)
   parser.add_argument("--device", default='xpu', type=str)
   parser.add_argument("--trace", default=None, type=str)
   args = parser.parse_args()
   return args
args = set_args()
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
if args.device == "xpu":
   import intel_extension_for_pytorch
   import oneccl_bindings_for_pytorch
   
if args.debug:
   logging.basicConfig(filename=args.output, level="DEBUG")
else:
   logging.basicConfig(filename=args.output, level="INFO")
logger = logging.getLogger(__name__)   
t2 = datetime.datetime.now()

elapsed = (t2 - GLOBAL_START_TIME).total_seconds()

master_port              = 2345
master_addr=os.environ["MASTER_ADDR"]

if (rank==0):
   logger.info(f"Imported all the libraries in {elapsed} seconds")
   logger.info(f"master_addr: {master_addr}")
   logger.info(f"{torch.__version__}")
   logger.info(f"{torch.__file__}")
   
os.environ["MASTER_PORT"]   = str(master_port)

t3 = datetime.datetime.now()
dist.init_process_group(backend = args.backend, init_method = 'env://', world_size = world_size, rank = rank, timeout = datetime.timedelta(seconds=120))
dist.barrier()
t4 = datetime.datetime.now()
elapsed = (t4 - t3).total_seconds()
if rank==0:
   logger.info(f"torch.distributed.init_process_group (time : {elapsed:.5f})")
dist_my_rank        = dist.get_rank()
dist_world_size     = dist.get_world_size()
assert(rank == dist_my_rank)
assert(world_size == dist_world_size)

def get_default_device():
   if args.device=="xpu":
      return torch.device(f"xpu:{local_rank}")
   else:
      return torch.device(f"cuda:{local_rank}")

def main():
   device  = get_default_device()
   ppn = world_size // args.pp
   my_layer = rank//ppn
   my_layer_local_rank = rank%ppn

   if (rank==0):
      logger.info(f"PP = {args.pp}, {ppn} per pp group")
      
   model = Model(num_layers = 1, pp = args.pp, rank = rank, world_size = world_size, device = device, logger=logger)
   
   if args.init_comm:
      t0 = time.time()
      model.comm_init()
      t1 = time.time()
      if rank ==0:
         logger.info(f"Time for init comm: {t1 - t0:.8f}")
   if args.init_comp:
      t0 = time.time()
      model.comp_init()
      t1 = time.time()
      if rank ==0:
         logger.info(f"Time for init comp: {t1 - t0:.8f}")

   dist.barrier()
   for it in range(args.niters):
      t0 = time.time()
      model.reset()
      model.forward()
      # sanity check
      assert(model.tensor[0]==model.my_pp_rank+1)
      t1 = time.time()
      model.backward()
      # sanity check      
      assert(model.tensor[0]==(args.pp + 2*(args.pp -  model.my_pp_rank)))
      dist.barrier()
      t2 = time.time()
      if rank ==0:
         logger.info(f"iter = {it}, fwd: {t1 - t0:.8f}, bwd: {t2 - t1:.8f}, total: {t2-t0:.8f}")
         
if __name__=='__main__':
   activities=[ProfilerActivity.CPU]
   if args.device == "xpu":
      activities.append(ProfilerActivity.XPU)
   else:
      activities.append(ProfilerActivity.CUDA)
   if args.trace is not None:
      with profile(activities=activities, record_shapes=True) as prof:
         main()
      prof.export_chrome_trace(f"{args.trace}-{rank}-of-{world_size}.json")
   else:
      main()
