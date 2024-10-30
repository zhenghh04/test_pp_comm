import datetime
import logging
import sys
import numpy as np
t1 = datetime.datetime.now()
import intel_extension_for_pytorch
import torch
import torch.nn.parallel
import os
import socket
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()
parser.add_argument("--pp", default=1, type=int)
parser.add_argument("--tp", default=1, type=int)
parser.add_argument("--backend", default='ccl', type=str)
parser.add_argument("--init", action='store_true')
parser.add_argument("--niters", default=10, type=int)
parser.add_argument("--debug", action='store_true')
parser.add_argument('--output', default="output.log", type=str)
parser.add_argument("--trace", default=None, type=str)
args = parser.parse_args()

if args.debug:
   logging.basicConfig(filename=args.output, level="DEBUG")
else:
   logging.basicConfig(filename=args.output, level="INFO")
logger = logging.getLogger(__name__)   
t2 = datetime.datetime.now()

import torch
elapsed = (t2 - t1).total_seconds()

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
master_port              = 2345

if (rank==0):
   logger.info(f"imported all the libraries in {elapsed} seconds")
   master_addr=os.environ["MASTER_ADDR"]
   logger.info(master_addr)
   logger.info(f"{torch.__version__}")
   logger.info(f"{torch.__file__}")
   

os.environ["MASTER_PORT"]   = str(master_port)

t3 = datetime.datetime.now()
from mpi4py import MPI
dist = MPI.COMM_WORLD
dist.barrier()
t4 = datetime.datetime.now()
elapsed = (t4 - t3).total_seconds()
if rank==0:
   logger.info(f"torch.distributed.init_process_group (time : {elapsed:.5f})")

def get_default_device():
   return torch.device(f"xpu:{local_rank}")

device  = get_default_device()
ppn = world_size // args.pp
my_layer = rank//ppn
my_layer_local_rank = rank%ppn


x_send = torch.ones(1024).to(device, non_blocking=True)
x_recv = torch.empty(1024).to(device, non_blocking=True)
if (rank==0):
   logger.info(f"PP = {args.pp}, {ppn} per pp group")
tensor = np.zeros(1024)
x = np.ones(1024)

def send_to_right():
   global tensor
   if my_layer != args.pp - 1:
      dist.send(tensor, dest=(rank+ppn)%world_size)
def recv_from_left():
   global tensor
   if my_layer != 0:
      tensor=dist.recv(source=(rank-ppn+world_size)%world_size)

def send_to_left():
   global tensor
   if my_layer != 0:   
      dist.send(tensor, dest=(rank-ppn+world_size)%world_size)
def recv_from_right():
   global tensor
   if my_layer != args.pp - 1:         
      tensor=dist.recv(source=(rank+ppn)%world_size)
      
def forward_pass_layer(L):
   global tensor
   assert(L<args.pp)
   if my_layer == L+1:
      tensor=dist.recv(source=rank-ppn)
      if my_layer_local_rank==0:      
         logger.debug(f"Forward {L+1} received: {tensor[0]}")
      tensor = tensor + x
      if my_layer_local_rank==0:      
         logger.debug(f"Forward {L+1} after compute: {tensor[0]}, added {x[0]}")
   elif my_layer == L:
      dist.send(tensor, dest=rank+ppn)
      if my_layer_local_rank==0:
         logger.debug(f"Forward {L} sent: {tensor[0]}")      
   dist.barrier()
def backward_pass_layer(L):
   global tensor
   assert(L>0)
   if my_layer == L-1:
      tensor=dist.recv(source=rank+ppn)
      if my_layer_local_rank==0:      
         logger.debug(f"Backward {L-1} received: {tensor[0]}")
      tensor = tensor + 2*x
      if my_layer_local_rank==0:      
         logger.debug(f"Backward {L+1} after compute: {tensor[0]}, added {2*x[0]}")
   elif my_layer == L:
      dist.send(tensor, dest=rank-ppn)
      if my_layer_local_rank==0:      
         logger.debug(f"Backward {L} sent: {tensor[0]}")
   dist.barrier()
   
def forward_pass():
   for L in range(0, args.pp-1):
      forward_pass_layer(L)
def backward_pass():
   for L in range(args.pp-1, 0, -1):
      backward_pass_layer(L)
   
def comm_init():
   dist.barrier()   
   t0 = time.time()
   if (my_layer)%2==1:
      logger.debug(f"L{my_layer}-r{my_layer_local_rank} recv issued at {time.time()-t0:.8f}")      
      recv_from_left()
      logger.debug(f"L{my_layer}-r{my_layer_local_rank} received at {time.time()-t0:.8f}")
   else:
      logger.debug(f"L{my_layer}-r{my_layer_local_rank} send issued at {time.time()-t0:.8f}")            
      send_to_right()
      logger.debug(f"L{my_layer}-r{my_layer_local_rank} sent at {time.time()-t0:.8f}")      
   dist.barrier()
   t1 = time.time()
   if rank==0:
      logger.info(f"F: 0->1, 2->3, 4-> ...: {t1-t0:.8f}")

   t0 = time.time()
   if (my_layer)%2==0:
      recv_from_right()
   else:
      send_to_left()
   if rank==0:
      logger.info(f"B: 0<-1, 2<-3, 4<- ...: {time.time() - t0:.8f}")
   dist.barrier()

   t0 = time.time()
   if (my_layer)%2==0:
      recv_from_left()
   else:
      send_to_right()
   dist.barrier()
   t1 = time.time()      
   if rank==0:
      logger.info(f"F: 0, 1->2, 3->4,  ...: {t1-t0:.8f}")      

   t0 = time.time()
   if (my_layer)%2==1:
      recv_from_right()
   else:
      send_to_left()
   dist.barrier()      
   if rank==0:
      logger.info(f"B: 0, 1<-2, 3<-4,  ...: {time.time() - t0:.8f}")

   
import time
def main():
   if args.init:
      t0 = time.time()
      comm_init()
      t1 = time.time()
      if rank ==0:
         logger.info(f"Time for init_comm: {t1 - t0:.8f}")
   for iter in range(args.niters):
      tensor = 0*x
      t0 = time.time()    
      forward_pass()
      dist.barrier()   
      # sanity check
      assert(tensor[0]==my_layer)
      t1 = time.time()
      backward_pass()
      # sanity check
      assert(tensor[0]==(args.pp - 1 + 2*(args.pp - 1 -  my_layer)))
      dist.barrier()
      t2 = time.time()
      if rank ==0:
         logger.info(f"iter = {iter}, fwd: {t1 - t0:.8f}, bwd: {t2 - t1:.8f}, total: {t2-t0:.8f}")

if __name__=='__main__':
   if args.trace is not None:
      activities = [ProfilerActivity.CPU]
      if args.device == "xpu":
         activities.append(ProfilerActivity.XPU)
      elif:
         activities.append(ProfilerActivity.GPU)
      with profile(activities=activities, record_shapes=True) as prof:
         logger.info(f"activities: {activities}")         
         main()
      prof.export_chrome_trace(args.trace)
   else:
      main()
