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
@click.option('--run-nrnmpi-init', type=bool, default=True)
@click.option('--use-subworlds', is_flag=True)
@click.option('--procs-per-worker', type=int, default=1)
@click.option('--sleep', type=float, default=0.)
def main(run_nrnmpi_init, use_subworlds, procs_per_worker, sleep):
    """

    :param run_nrnmpi_init: bool
    :param use_subworlds: bool
    :param procs_per_worker: int
    :param sleep: float
    """
    time.sleep(sleep)

    if run_nrnmpi_init:
        try:
            h.nrnmpi_init()
            print('diagnoise_h_quit: getting past h.nrnmpi_init()')
            sys.stdout.flush()
            time.sleep(1.)
        except:
            print('diagnoise_h_quit: problem calling h.nrnmpi_init(); may not be defined in this version of NEURON')
            sys.stdout.flush()
            time.sleep(1.)
    else:
        print('diagnoise_h_quit: h.nrnmpi_init() not executed')
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

    if use_subworlds:
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


if __name__ == '__main__':
    main(standalone_mode=False)
    print('calling h_quit')
    sys.stdout.flush()
    time.sleep(1.)
    h.quit()
    print('Got past h_quit')
    sys.stdout.flush()
    time.sleep(1.)
