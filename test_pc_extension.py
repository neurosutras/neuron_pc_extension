from pc_extension import *
import click


context = Context()


def test(first, second, third=None):
    """
    
    :param first:
    :param second:
    :param third:
    :return:
    """
    if 'count' not in context():
        context.count = 0
    # 20180221: Troubleshooting apply on Cori
    if context.interface.global_rank == 0:
        print 'master process executing test with count: %i' % context.count
    context.update(locals())
    context.count += 1
    time.sleep(0.2)
    return 'pid: %i, args: %s, count: %i' % (os.getpid(), str([first, second, third]), context.count)


def init_worker():
    """

    :return:
    """
    context.interface.start(disp=True)
    context.interface.ensure_controller()
    return context.interface.global_rank


@click.command()
@click.option("--procs-per-worker", type=int, default=1)
def main(procs_per_worker):
    """

    :param procs_per_worker: int
    """
    context.interface = ParallelContextInterface(procs_per_worker=procs_per_worker)
    result1 = context.interface.get('context.interface.global_rank')
    if context.interface.global_rank == 0:
        print 'before interface start: %i/%i total processes detected\n' % \
              (len(set(result1)), context.interface.global_size)
    time.sleep(1.)
    sys.stdout.flush()

    result2 = context.interface.apply(init_worker)
    time.sleep(1.)
    sys.stdout.flush()
    if len(result2) == 1 and result2[0] == 0:
        print 'after interface start: just master process returned from init_worker\n'
    else:
        print 'after interface start: something went wrong; %i processes returned from init_worker\n' % len(result2)
    sys.stdout.flush()
    time.sleep(1.)

    start1 = 0
    end1 = start1 + int(context.interface.global_size)
    start2 = end1
    end2 = start2 + int(context.interface.global_size)
    print ': context.interface.map_sync(test, range(%i, %i), range(%i, %i))' % (start1, end1, start2, end2)
    pprint.pprint(context.interface.map_sync(test, range(start1, end1), range(start2, end2)))
    print '\n'
    sys.stdout.flush()
    time.sleep(1.)

    print ': context.interface.map_async(test, range(%i, %i), range(%i, %i))' % (start1,end1, start2, end2)
    result3 = context.interface.map_async(test, range(start1, end1),range(start2, end2))
    while not result3.ready():
        time.sleep(0.1)
    result3 = result3.get()
    pprint.pprint(result3)
    print '\n'
    sys.stdout.flush()
    time.sleep(1.)

    print ': This is where the problems lie:'
    print ': context.interface.apply(test, 1, 2, third=3)'
    pprint.pprint(context.interface.apply(test, 1, 2, third=3))
    print '\n'
    sys.stdout.flush()
    time.sleep(1.)

    print 'context.interface.get(\'context.interface.global_rank\')'
    result4 = context.interface.get('context.interface.global_rank')
    print 'before interface stop: %i/%i workers detected\n' % (len(set(result4)), context.interface.num_workers)
    sys.stdout.flush()
    time.sleep(1.)
    context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)
