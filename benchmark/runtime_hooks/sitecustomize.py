"""Opt-in title inherited by trainer subprocesses in this benchmark only."""
import os
if os.environ.get('BASELINE_PROCESS_TITLE')=='VLLM::Enginecore':
    from setproctitle import setproctitle
    setproctitle('VLLM::Enginecore')
