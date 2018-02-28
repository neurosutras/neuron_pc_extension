'''
Usage:
from submit_impl_of_context import pccontext

def context(arg):
  ...

# for any pc.subworld organization
#after pc.runworker()
#execute context(arg) on all nhost_world ranks except 0
pccontext(context, arg)

'''

'''
Implementation of what pc.context should be via use of pc.submit. The problem
with pc.context is that it does not execute on any worker of subworld 0.
by submitting pc.nhost_bbs jobs with a context_callable, context pair
of args and arranging them to execute one per pc.id_bbs
we can get the effect of what pc.context should be. Note that
id_world==0 does NOT execute context_callable(context)
'''
from mpi4py import MPI
from neuron import h
import sys


pc = h.ParallelContext()


def _context(context, arg):
    if (int(pc.id_world()) > 0):
        context(arg)
    else:
        print ("master entered _context\r")
        sys.stdout.flush()
    if (int(pc.id()) == 0):  # increment context count
        pc.master_works_on_jobs(0)
        pc.take("context")
        pc.master_works_on_jobs(1)
        i = pc.upkscalar()
        pc.post("context", i + 1)
        while True:
            pc.take("context")
            i = pc.upkscalar()
            pc.post("context", i)
            time.sleep(0.1)
            if i == nhost_bbs:
                return  # nhost_bbs distinct ranks executed _context


def pccontext(context, arg):  # working version of pc.context(context, arg)
    pc.post("context", 0)
    for i in range(int(pc.nhost_bbs())):
        pc.submit(_context, context, arg)
    while pc.working():
        pass
    pc.take("context")


if __name__ == "__main__":
    import time

    pc.subworlds(2)
    nhost_world = int(pc.nhost_world())
    id_world = int(pc.id_world())
    nhost_bbs = int(pc.nhost_bbs())
    id_bbs = int(pc.id_bbs())
    nhost = int(pc.nhost())
    id = int(pc.id())


    def f(arg):
        print ("arg=%d nhost_world=%d id_world=%d nhost_bbs=%d id_bbs=%d nhost=%d id=%d\r" %
               (arg, nhost_world, id_world, nhost_bbs, id_bbs, nhost, id))


    f(1)
    sys.stdout.flush()
    time.sleep(1.)  # enough time to print

    pc.runworker()
    print ("after runworker\r")
    sys.stdout.flush()
    time.sleep(1.)  # enough time to print

    pccontext(f, 2)
    sys.stdout.flush()
    time.sleep(1.)  # enough time to print

    f(3)  # rank 0 of the master subworld

    for i in range(1):  # time to print and
        pc.post("wait")  # bulletin board to communicate
        time.sleep(.1)
        pc.take("wait")

    pc.done()
    h.quit()
