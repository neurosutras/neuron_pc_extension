import sys
import os
import time
import click


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--import-mpi4py', type=int, default=0)
@click.option('--h-quit', is_flag=True)
@click.option('--procs-per-worker', type=int, default=1)
@click.option('--sleep', type=float, default=0.)
def main(import_mpi4py, h_quit, procs_per_worker, sleep):
    """

    :param import_mpi4py: int
    :param h_quit: bool
    :param procs_per_worker: int
    """
    time.sleep(sleep)
    if import_mpi4py == 1:
        order = 'before'
        from mpi4py import MPI
        print('test_mpiguard: getting past import mpi4py')
        sys.stdout.flush()
        time.sleep(1.)

    from neuron import h
    print('test_mpiguard: getting past from neuron import h')
    sys.stdout.flush()
    time.sleep(1.)

    try:
        h.nrnmpi_init()
        print('test_mpiguard: getting past h.nrnmpi_init()')
        sys.stdout.flush()
        time.sleep(1.)
    except:
        print('test_mpiguard: problem calling h.nrnmpi_init(); may not be defined for this version of neuron')
        sys.stdout.flush()
        time.sleep(1.)

    if import_mpi4py == 2:
        order = 'after'
        from mpi4py import MPI

    if import_mpi4py > 0:
        comm = MPI.COMM_WORLD

    pc = h.ParallelContext()
    pc.subworlds(procs_per_worker)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())

    if import_mpi4py > 0:
        print('test_mpiguard: mpi4py imported %s neuron: process id: %i; global rank: %i / %i; local rank: %i / %i; '
              'comm.rank: %i; comm.size: %i' %
              (order, os.getpid(), global_rank, global_size, rank, size, comm.rank, comm.size))
    else:
        print('test_mpiguard: mpi4py not imported: process id: %i; global rank: %i / %i; local rank: %i / %i' %
              (os.getpid(), global_rank, global_size, rank, size))
    sys.stdout.flush()
    time.sleep(1.)

    pc.runworker()

    print('test_mpiguard: got past pc.runworker()')
    sys.stdout.flush()
    time.sleep(1.)

    # catch workers escaping from runworker loop
    if global_rank != 0:
        print('test_mpiguard: global_rank: %i escaped from the pc.runworker loop')
        sys.stdout.flush()
        time.sleep(1.)
        os._exit(1)

    pc.done()
    if h_quit:
        print('calling h_quit')
        sys.stdout.flush()
        time.sleep(1.)
        h.quit()
    else:
        print('trying to exit without calling h_quit')
        sys.stdout.flush()
        time.sleep(1.)


if __name__ == '__main__':
    main(standalone_mode=False)
