import sys
import os
import time
from mpi4py import MPI
print('diagnoise_h_quit: getting past import mpi4py')
sys.stdout.flush()
time.sleep(1.)
from neuron import h
print('diagnoise_h_quit: getting past from neuron import h')
sys.stdout.flush()
time.sleep(1.)

comm = MPI.COMM_WORLD

h.nrnmpi_init()
print('diagnoise_h_quit: getting past h.nrnmpi_init()')
sys.stdout.flush()
time.sleep(1.)

pc = h.ParallelContext()
pc.subworlds(1)
global_rank = int(pc.id_world())
global_size = int(pc.nhost_world())
rank = int(pc.id())
size = int(pc.nhost())
print('diagnoise_h_quit: process id: %i; global rank: %i / %i; local rank: %i / %i; mpi comm rank: %i / %i' %
      (os.getpid(), global_rank, global_size, rank, size, comm.rank, comm.size))
sys.stdout.flush()
time.sleep(1.)


pc.runworker()
print('diagnoise_h_quit: got past pc.runworker()')
sys.stdout.flush()
time.sleep(1.)

# catch workers escaping from runworker loop
if global_rank != 0:
    print('diagnoise_h_quit: global_rank: %i escaped from the pc.runworker loop')
    sys.stdout.flush()
    time.sleep(1.)
    os._exit(1)

pc.done()
print('diagnoise_h_quit: got past pc.done()')
sys.stdout.flush()
time.sleep(1.)
# os._exit(1)

print('diagnoise_h_quit: MPI.Is_finalized(): %s' % str(MPI.Is_finalized()))
sys.stdout.flush()
time.sleep(1.)

print('diagnoise_h_quit: calling h_quit()')
sys.stdout.flush()
time.sleep(1.)
h.quit()
print('diagnoise_h_quit: got past h_quit()')
sys.stdout.flush()
time.sleep(1.)
