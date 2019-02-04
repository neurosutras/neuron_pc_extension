import sys
import os
import time
import click
try:
    from mpi4py import MPI
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    raise ImportError('test_mpi_futures_submit_limits: problem with from mpi4py.futures')


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """
    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is None:
            namespace_dict = {}
        namespace_dict.update(kwargs)
        for key, value in namespace_dict.iteritems():
            setattr(self, key, value)

    def __call__(self):
        return self.__dict__


context = Context()
context.comm = MPI.COMM_WORLD
context.rank = context.comm.Get_rank()
context.size = context.comm.Get_size()

print 'test_mpi_futures_submit_limits: process id: %i; global rank: %i / %i' % \
          (os.getpid(), context.rank, context.size)
sys.stdout.flush()
time.sleep(1.)


def test(val):
    return val


def mpi_map(func, *sequences):
    """
    This method uses the mpi4py.futures bulletin board to implement a map operation. Returns results as a list
    in the same order as the specified sequences.
    :param func: callable
    :param sequences: list
    :return: list
    """
    if not sequences:
        return None

    futures = []
    for args in zip(*sequences):
        futures.append(context.executor.submit(func, *args))
        context.task_counter += 1

    return [future.result() for future in futures]


@click.command()
@click.option("--block-size", type=int, default=None)
@click.option("--task-limit", type=int, default=3000000)
def main(block_size, task_limit):
    """
    Executes blocks of tasks using MPIPoolExecutor.submit until a task limit.
    :param block_size: int
    :param task_limit: int
    """
    context.executor = MPIPoolExecutor()

    context.task_counter = 0

    # catch workers escaping from executor
    if context.rank != 0:
        os._exit(1)

    if block_size is None:
        block_size = context.size
    while context.task_counter < task_limit:
        result = mpi_map(test, range(block_size))
        print 'Completed tasks count: %i' % context.task_counter
        sys.stdout.flush()
        time.sleep(0.1)

    context.executor.shutdown()



if __name__ == '__main__':
    main(standalone_mode=False)
