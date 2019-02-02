import sys
import os
import time
import click
try:
    from mpi4py import MPI
    from neuron import h
except ImportError:
    raise ImportError('pc_extension: problem with importing neuron')


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


def test(val):
    return val


def pc_map(pc, func, *sequences):
    """
    ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
    processes, and returning the results in the same order as the specified sequence. This method implements a
    synchronous (blocking) map operation. Returns results as a list in the same order as the specified sequences.
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
        key = context.task_counter
        context.task_counter += 1
        pc.submit(key, func, *args)
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


@click.command()
@click.option("--block-size", type=int, default=None)
@click.option("--task-limit", type=int, default=100)
def main(block_size, task_limit):
    """
    Execute blocks of tasks using pc.submit until a task limit.
    :param block_size: int
    :param task_limit: int
    """

    pc = h.ParallelContext()
    pc.subworlds(1)
    global_rank = int(pc.id_world())
    global_size = int(pc.nhost_world())
    rank = int(pc.id())
    size = int(pc.nhost())
    task_counter = 0

    print 'test_pc_submit_limits: process id: %i; global rank: %i / %i; local rank: %i / %i' % \
          (os.getpid(), global_rank, global_size, rank, size)
    sys.stdout.flush()
    time.sleep(1.)

    context.update(locals())
    pc.runworker()

    # catch workers escaping from runworker loop
    if global_rank != 0:
        os._exit(1)

    if block_size is None:
        block_size = global_size
    while context.task_counter < task_limit:
        result = pc_map(pc, test, range(block_size))
        print 'Completed tasks count: %i' % context.task_counter
        sys.stdout.flush()
        time.sleep(0.1)

    pc.done()


if __name__ == '__main__':
    main(standalone_mode=False)
