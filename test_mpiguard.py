import sys
import os
import time
import click


def test(val):
    return val


def pc_map(pc, func, *sequences):
    """
    This method uses the NEURON ParallelContext bulletin board to implements a map operation. Returns results as a list
    in the same order as the specified sequences.
    :param pc: :class:'h.ParallelContext'
    :param func: callable
    :param sequences: list
    :return: list
    """
    if not sequences:
        return None
    keys = []
    results_dict = dict()

    for args in zip(*sequences):
        # key = context.task_counter
        context.task_counter += 1
        key = pc.submit(func, *args)
        keys.append(key)

    remaining_keys = list(keys)
    while len(remaining_keys) > 0 and pc.working():
        key = int(pc.userid())
        results_dict[key] = pc.pyret()
        remaining_keys.remove(key)
    try:
        return [results_dict.pop(key) for key in keys]
    except KeyError:
        raise KeyError('pc_map: all jobs have completed, but not all requested keys were found')


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option('--import-mpi4py', type=int, default=0)
@click.option('--h-quit', is_flag=True)
@click.option('--procs-per-worker', type=int, default=1)
def main(import_mpi4py, h_quit, procs_per_worker):
    """

    :param import_mpi4py: int
    :param h_quit: bool
    :param procs_per_worker: int
    """
    if import_mpi4py == 1:
        order = 'before'
    elif import_mpi4py == 2:
        order = 'after'

    if import_mpi4py == 1:
        from mpi4py import MPI
    from neuron import h
    h.nrnmpi_init()
    if import_mpi4py == 2:
        from mpi4py import MPI
    if import_mpi4py > 0:
        comm = MPI.COMM_WORLD
        print('test_mpiguard: mpi4py imported %s neuron; got past imports' % order)
    else:
        print('test_mpiguard: mpi4py not imported; got past imports')
    sys.stdout.flush()
    time.sleep(1.)

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
