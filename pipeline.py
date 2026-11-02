"""Foreground monitor: success means all queued trainings and reports completed."""
import argparse
import fcntl
import time
from common import ROOT, read
from campaign import database, launch, worker_alive
from report import main as report


def completion_lines(workers, offsets):
    """Read only completion/failure lines; never echo per-step task logs."""
    from pathlib import Path
    for worker in workers:
        path = Path(worker['log'])
        if not path.exists():
            continue
        with path.open() as log:
            log.seek(offsets.get(str(path), 0))
            while True:
                start = log.tell()
                line = log.readline()
                if not line or not line.endswith('\n'):
                    log.seek(start)
                    break
                if line.startswith(('DONE ', 'FAILED ')):
                    print(line.rstrip(), flush=True)
            offsets[str(path)] = log.tell()


def run(args):
    with (ROOT / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = database()
        workers = read(ROOT / 'launch.json') if (ROOT / 'launch.json').exists() else []
        live = [w for w in workers if worker_alive(w['pid'])]
        if not live:
            # Interrupted tasks resume from their own verified checkpoints.
            db.execute("update tasks set state='pending',worker='' where state='running'")
            if args.retry_failed:
                db.execute("update tasks set state='pending',error='' where state='failed'")
            db.commit()
        db.close()
        # Do not replay completion messages from earlier invocations.
        from pathlib import Path
        offsets = {w['log']: Path(w['log']).stat().st_size for w in workers if Path(w['log']).exists()}
        offsets.update({str(p):p.stat().st_size for p in (ROOT/'logs').glob('worker_gpu*.log')})
        try:
            previous = None
            while True:
                completion_lines(workers, offsets)
                db = database()
                counts = dict(db.execute('select state,count(*) from tasks group by state').fetchall())
                db.close()
                if counts != previous:
                    report(); previous = counts
                if not counts.get('pending', 0) and not counts.get('running', 0):
                    # The worker marks the DB complete immediately before its
                    # final print. Let it flush and exit before draining logs.
                    if any(worker_alive(w['pid']) for w in workers):
                        time.sleep(args.poll_seconds)
                        continue
                    if counts.get('failed', 0):
                        raise RuntimeError('Tasks failed; see per-task logs; fix and rerun with --retry-failed')
                    print('COMPLETED:', report(), flush=True)
                    return
                live = [w for w in workers if worker_alive(w['pid'])]
                if not live:
                    if counts.get('running', 0):
                        raise RuntimeError('Workers stopped unexpectedly; rerun to resume interrupted tasks')
                    if counts.get('failed', 0):
                        raise RuntimeError('Queue blocked by failed tasks; see logs and rerun with --retry-failed')
                    launch(args)
                    workers = read(ROOT / 'launch.json')
                    # New workers append to their existing logs. Start at the
                    # previous end so completed models are printed once only.
                    for w in workers:
                        offsets.setdefault(w['log'],0)
                time.sleep(args.poll_seconds)
        finally:
            completion_lines(workers, offsets)
            report()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', default='0')
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--poll-seconds', type=float, default=15)
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    if a.workers_per_gpu < 1 or a.poll_seconds <= 0 or not a.gpus.strip():
        p.error('GPU list, positive worker count and positive polling interval are required')
    run(a)
