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
import click

comm = MPI.COMM_WORLD


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--use-subworlds', is_flag=True)
@click.option('--call-pc-done', is_flag=True)
@click.option('--procs-per-worker', type=int, default=1)
def main(use_subworlds, call_pc_done, procs_per_worker):
    """

    :param use_subworlds: bool
    :param call_pc_done: bool
    :param procs_per_worker: int
    """
    h.nrnmpi_init()
    print('diagnoise_h_quit: getting past h.nrnmpi_init()')
    sys.stdout.flush()
    time.sleep(1.)

    pc = h.ParallelContext()
    if use_subworlds:
        pc.subworlds(procs_per_worker)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    print('diagnoise_h_quit: process id: %i; global rank: %i / %i; local rank: %i / %i' %
          (os.getpid(), global_rank, global_size, rank, size))
    sys.stdout.flush()
    time.sleep(1.)

    print('diagnoise_h_quit: process id: %i; mpi rank: %i / %i' % (os.getpid(), comm.rank, comm.size))
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

    if call_pc_done:
        pc.done()

    print('calling h_quit')
    sys.stdout.flush()
    time.sleep(1.)
    h.quit()
    print('Got past h_quit')
    sys.stdout.flush()
    time.sleep(1.)

if __name__ == '__main__':
    main(standalone_mode=False)
