import datetime
import logging
import sys

t1 = datetime.datetime.now()
import intel_extension_for_pytorch
import torch
import torch.nn.parallel
import os
import socket
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pp", default=1, type=int)
parser.add_argument("--tp", default=1, type=int)
parser.add_argument("--backend", default='ccl', type=str)
parser.add_argument("--init", action='store_true')
parser.add_argument("--niters", default=10, type=int)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()


t2 = datetime.datetime.now()

import torch
elapsed = (t2 - t1).total_seconds()

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
master_port              = 2345
if args.debug:
   logging.basicConfig(filename='test_comm.log', level=logging.DEBUG)
else:
   logging.basicConfig(filename='test_comm.log', level=logging.INFO)
logger = logging.getLogger()
#logger = logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

if (rank==0):
   logger.info(f"imported all the libraries in {elapsed} seconds")
   master_addr=os.environ["MASTER_ADDR"]
   logger.info(master_addr)
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

def get_default_device():
   return torch.device(f"xpu:{local_rank}")

device  = get_default_device()
ppn = world_size // args.pp

x = torch.ones(1024).to(device, non_blocking=True)
x_send = torch.ones(1024).to(device, non_blocking=True)
x_recv = torch.empty(1024).to(device, non_blocking=True)
if (rank==0):
   logger.info(f"PP = {args.pp}, {ppn} per pp group")

def forward_pass_concurrent():
   # getting data from previous pp rank
   if rank >= ppn:
      dist.recv(tensor=x_recv, src=rank-ppn)
   # sending data to next pp rank
   if rank < world_size - ppn:
      dist.send(tensor=x_send, dst=rank+ppn)
def backward_pass_concurrent():
   if rank < world_size - ppn:
      dist.recv(tensor=x_recv, src=rank+ppn)
   if rank >= ppn:
      dist.send(tensor=x_send, dst=rank-ppn)

tensor = None
if (rank//ppn==0):
   tensor = 0*x
else:
   tensor = torch.empty(1024).to(device, non_blocking=True)
   
def forward_pass_layer(L):
   global tensor
   assert(L<args.pp)
   if rank//ppn == L+1:
      tensor = torch.empty(1024).to(device, non_blocking=True)
      dist.recv(tensor=tensor, src=rank-ppn)
      if rank%ppn==0:      
         logger.debug(f"Forward {L+1} received: {tensor[0]}")
      tensor = tensor + x
      if rank%ppn==0:      
         logger.debug(f"Forward {L+1} after compute: {tensor[0]}, added {x[0]}")
   elif rank//ppn == L:
      dist.send(tensor=tensor, dst=rank+ppn)
      if rank%ppn==0:
         logger.debug(f"Forward {L} sent: {tensor[0]}")      
   #dist.barrier()
   
def backward_pass_layer(L):
   global tensor
   assert(L>0)
   if rank//ppn == L-1:
      tensor = torch.empty(1024).to(device, non_blocking=True)      
      dist.recv(tensor=tensor, src=rank+ppn)
      if rank%ppn==0:      
         logger.debug(f"Backward {L-1} received: {tensor[0]}")
      tensor = tensor + 2*x
      if rank%ppn==0:      
         logger.debug(f"Forward {L+1} after compute: {tensor[0]}, added {2*x[0]}")
   elif rank//ppn == L:
      dist.send(tensor=tensor, dst=rank-ppn)
      if rank%ppn==0:      
         logger.debug(f"Backward {L} sent: {tensor[0]}")

def forward_pass():
   for L in range(0, args.pp-1):
      forward_pass_layer(L)
   dist.barrier()
def backward_pass():
   for L in range(args.pp-1, 0, -1):
      backward_pass_layer(L)
   dist.barrier()
   
def comm_init():
   forward_pass_concurrent()
   backward_pass_concurrent()
   dist.barrier()
   
rank = dist.get_rank()
size = dist.get_world_size()
import time
if args.init:
   t0 = time.time()
   comm_init()
   t1 = time.time()
   if rank ==0:
      logger.info(f"Time for init_comm: {t1 - t0:.8f}")
   
for iter in range(args.niters):
   t0 = time.time()    
   forward_pass()
   t1 = time.time()
   backward_pass()
   t2 = time.time()
   if rank ==0:
      logger.info(f"iter = {iter}, fwd: {t1 - t0:.8f}, bwd: {t2 - t1:.8f}")
