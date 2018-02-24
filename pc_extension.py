"""
neuron_pc_extension

Classes and methods to provide an interface to extend the NEURON ParallelContext bulletin board for flexible
nested parallel computations.
"""
__author__ = 'Aaron D. Milstein'
import sys
import os
import time
import pprint
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


class AsyncResultWrapper(object):
    """
    When ready(), get() returns results as a list in the same order as submission.
    """
    def __init__(self, interface, keys):
        """

        :param interface: :class: 'ParallelContextInterface'
        :param keys: list
        """
        self.interface = interface
        self.keys = keys
        self._ready = False

    def ready(self):
        """

        :return: bool
        """
        if self.interface.pc.working():
            key = int(self.interface.pc.userid())
            self.interface.collected[key] = self.interface.pc.pyret()
        else:
            self._ready = True
            return True
        if all(key in self.interface.collected for key in self.keys):
            self._ready = True
            return True
        else:
            return False

    def get(self):
        """
        Returns None until all results have completed, then returns a list of results in the order of original
        submission.
        :return: list
        """
        if self._ready or self.ready():
            try:
                return [self.interface.collected.pop(key) for key in self.keys]
            except KeyError:
                raise KeyError('AsyncResultWrapper: all jobs have completed, but not all requested keys were found')
        else:
            return None


class ParallelContextInterface(object):
    """
    Class provides an interface to extend the NEURON ParallelContext bulletin board for flexible nested parallel
    computations.
    """
    def __init__(self, procs_per_worker=1):
        """

        :param procs_per_worker: int
        """
        self.global_comm = MPI.COMM_WORLD
        self.procs_per_worker = procs_per_worker
        self.pc = h.ParallelContext()
        self.pc.subworlds(procs_per_worker)
        self.global_rank = int(self.pc.id_world())
        self.global_size = int(self.pc.nhost_world())
        self.rank = int(self.pc.id())
        self.size = int(self.pc.nhost())
        global_ranks = [self.global_rank] * self.size
        global_ranks = self.pc.py_alltoall(global_ranks)
        group = self.global_comm.Get_group()
        sub_group = group.Incl(global_ranks)
        self.comm = self.global_comm.Create(sub_group)
        self.worker_id = self.comm.bcast(int(self.pc.id_bbs()), root=0)
        self.num_workers = self.comm.bcast(int(self.pc.nhost_bbs()), root=0)
        # 'collected' dict acts as a temporary storage container on the master process for results retrieved from
        # the ParallelContext bulletin board.
        self.collected = {}
        assert self.rank == self.comm.rank and self.global_rank == self.global_comm.rank and \
               self.global_comm.size / self.procs_per_worker == self.num_workers, \
            'pc_extension: pc.ids do not match MPI ranks'
        self._running = False
        self.map = self.map_sync
        self.apply = self.apply_sync
        self.apply_counter = 0

    def print_info(self):
        print 'pc_extension: process id: %i; global rank: %i / %i; local rank: %i / %i; worker id: %i / %i' % \
              (os.getpid(), self.global_rank, self.global_size, self.comm.rank, self.comm.size, self.worker_id,
               self.num_workers)
        time.sleep(0.1)

    def master_wait_for_all_workers(self, key):
        """
        Prevents the master from taking a job, or even checking for job results, until all workers have completed the
        operation associated with the specified key.
        :param key: int or str
        """
        if self.global_rank != 0:
            raise ValueError('pc_extension: global_rank: %i; should not have entered master_wait_for_all_workers' %
                             self.global_rank)
        global_count = 0
        iter_count = 0
        while global_count < self.num_workers - 1:
            print 'global_rank: %i; inside the master_wait_for_all_workers loop; %i iterations\r' % \
                  (self.global_rank, iter_count)
            sys.stdout.flush()
            self.pc.take(key)
            count = self.pc.upkscalar()
            global_count = count
            self.pc.post(key, count)
            time.sleep(0.1)
            iter_count += 1
        print 'global_rank: %i; exiting master_wait_for_all_workers; %i iterations\r' % \
              (self.global_rank, iter_count - 1)
        sys.stdout.flush()
        return

    def wait_for_all_workers(self, key):
        """
        Prevents any worker from returning until all workers have completed the operation associated with the specified
        key.
        :param key: int or str
        """
        global_ranks = [int(self.procs_per_worker * i) for i in xrange(self.num_workers)]
        incremented = False
        iter_count = 0
        global_count = 0
        if self.global_rank in global_ranks:
            while global_count < self.num_workers:
                for this_rank in global_ranks:
                    if self.global_rank == this_rank:
                        print 'global_rank: %i; local_rank: %i; inside the wait_for_all_workers loop; %i ' \
                              'iterations\r' % (self.global_rank, self.rank, iter_count)
                        sys.stdout.flush()
                        self.pc.take(key)
                        count = self.pc.upkscalar()
                        if not incremented:
                            count += 1
                            incremented = True
                        global_count = count
                        self.pc.post(key, count)
                        # time.sleep(0.1)
                iter_count += 1
            print 'global_rank: %i; local_rank: %i; exiting wait_for_all_workers; %i iterations\r' % \
                  (self.global_rank, self.rank, iter_count - 1)
            sys.stdout.flush()
        else:
            print 'global_rank: %i; local_rank: %i; is just along for the ride\r' % (self.global_rank, self.rank)
            sys.stdout.flush()
        return

    def apply_sync(self, func, *args, **kwargs):
        """
        ParallelContext lacks a native method to guarantee execution of a function on all workers. This method
        implements a synchronous (blocking) apply operation that accepts **kwargs and returns values collected from each
        worker.
        :param func: callable
        :param args: list
        :param kwargs: dict
        :return: dynamic
        """
        if self._running:
            apply_key = str(self.apply_counter)
            self.apply_counter += 1
            self.pc.post(apply_key, 0)
            keys = []
            for i in xrange(self.num_workers):
                keys.append(int(self.pc.submit(pc_apply_wrapper, func, apply_key, args, kwargs)))
            if self.global_rank == 0:
                self.master_wait_for_all_workers(apply_key)
            results = self.collect_results(keys)
            self.pc.take(apply_key)
            sys.stdout.flush()
            return [results[key] for key in keys]
        else:
            result = func(*args, **kwargs)
            sys.stdout.flush()
            if not self._running:
                results = self.global_comm.gather(result, root=0)
                if self.global_rank == 0:
                    return results
            elif result is None:
                return
            else:
                return [result]
    
    def collect_results(self, keys=None):
        """
        If no keys are specified, this method is a blocking operation that waits until all previously submitted jobs 
        have been completed, retrieves all results from the bulletin board, and stores them in the 'collected' dict in 
        on the master process, indexed by their submission key.
        If a list of keys is provided, collect_results first checks if the results have already been placed in the
        'collected' dict, and otherwise blocks until all requested results are available. Results retrieved from the
        bulletin board that were not requested are left in the 'collected' dict.
        :param keys: list
        :return: dict
        """
        if keys is None:
            while self.pc.working():
                key = int(self.pc.userid())
                self.collected[key] = self.pc.pyret()
                # time.sleep(0.1)
            keys = self.collected.keys()
            return {key: self.collected.pop(key) for key in keys}
        else:
            pending_keys = [key for key in keys if key not in self.collected]
            while self.pc.working():
                key = int(self.pc.userid())
                self.collected[key] = self.pc.pyret()
                if key in pending_keys:
                    pending_keys.remove(key)
                if not pending_keys:
                    break
                # time.sleep(0.1)
            return {key: self.collected.pop(key) for key in keys if key in self.collected}

    def map_sync(self, func, *sequences):
        """
        ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
        processes, and returning the results in the same order as the specified sequence. This method implements a
        synchronous (blocking) map operation. Returns results as a list in the same order as the specified sequences.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        keys = []
        for args in zip(*sequences):
            key = int(self.pc.submit(func, *args))
            keys.append(key)
        results = self.collect_results(keys)
        return [results[key] for key in keys]

    def map_async(self, func, *sequences):
        """
        ParallelContext lacks a native method to apply a function to sequences of arguments, using all available
        processes, and returning the results in the same order as the specified sequence. This method implements an
        asynchronous (non-blocking) map operation. Returns a PCAsyncResult object to track progress of the submitted
        jobs.
        :param func: callable
        :param sequences: list
        :return: list
        """
        if not sequences:
            return None
        keys = []
        for args in zip(*sequences):
            key = int(self.pc.submit(func, *args))
            keys.append(key)
        return AsyncResultWrapper(self, keys)

    def get(self, object_name):
        """
        ParallelContext lacks a native method to get the value of an object from all workers. This method implements a
        synchronous (blocking) pull operation.
        :param object_name: str
        :return: dynamic
        """
        return self.apply_sync(find_nested_object, object_name)

    def start(self, disp=False):
        if disp:
            self.print_info()
        self._running = True
        self.pc.runworker()

    def stop(self):
        self.pc.done()
        self._running = False

    def ensure_controller(self):
        """
        Exceptions in python on an MPI rank are not enough to end a job. Strange behavior results when an unhandled
        Exception occurs on an MPI rank while running a neuron.h.ParallelContext.runworker() loop. This method will
        hard exit python if executed by any rank other than the master.
        """
        if self.global_rank != 0:
            os._exit(1)


def pc_apply_wrapper(func, key, args, kwargs):
    """
    Methods internal to an instance of a class cannot be pickled and submitted to the neuron.h.ParallelContext bulletin 
    board for remote execution. As long as a module executes 'from nested.parallel import *', this method can be
    submitted to the bulletin board for remote execution, and prevents any worker from returning until all workers have
    applied the specified function.
    :param func: callable
    :param key: int or str
    :param args: list
    :param kwargs: dict
    :return: dynamic
    """
    result = func(*args, **kwargs)
    sys.stdout.flush()
    interface = pc_find_interface()
    interface.wait_for_all_workers(key)
    return result


def pc_find_interface():
    """
    ParallelContextInterface apply and get operations require a remote instance of ParallelContextInterface. This method
    attemps to find it in the remote __main__ namespace, or in a Context object therein.
    :return: :class:'ParallelContextInterface'
    """
    interface = None
    try:
        module = sys.modules['__main__']
        for item_name in dir(module):
            if isinstance(getattr(module, item_name), ParallelContextInterface):
                interface = getattr(module, item_name)
                break
        if interface is None:
            context = None
            for item_name in dir(module):
                if isinstance(getattr(module, item_name), Context):
                    context = getattr(module, item_name)
                    break
            if context is not None:
                for item_name in context():
                    if isinstance(getattr(context, item_name), ParallelContextInterface):
                        interface = getattr(context, item_name)
                        break
            if interface is None:
                raise Exception
        return interface
    except Exception:
        raise Exception('pc_extension: remote instance of ParallelContextInterface not found in '
                        'the remote __main__ namespace')


def find_nested_object(object_name):
    """
    This method attemps to find the object corresponding to the provided object_name (str) in the __main__ namespace.
    Tolerates objects nested in other objects.
    :param object_name: str
    :return: dynamic
    """
    this_object = None
    try:
        module = sys.modules['__main__']
        for this_object_name in object_name.split('.'):
            if this_object is None:
                this_object = getattr(module, this_object_name)
            else:
                this_object = getattr(this_object, this_object_name)
        if this_object is None:
            raise Exception
        return this_object
    except Exception:
        raise Exception('pc_extension: object: %s not found in remote __main__ namespace' % object_name)
