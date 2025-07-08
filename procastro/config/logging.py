import gzip
import logging as _log
import shutil
from pathlib import Path

from procastro.config import config_user

__all__ = ['pa_logger']

logging_config = config_user("logging", read_default=True)

pa_logger = _log.getLogger('procastro')

file_logging = logging_config["file_logging"]
if file_logging['log_to_file']:
    logfile = Path(file_logging['log_directory'])/'procastro.log'
    log_max_size = file_logging['log_file_max_size']

    # once it reaches max size, it compresses the current logfile and starts a new one
    if logfile.exists() and logfile.stat().st_size > log_max_size:
        with open(logfile, 'rb') as f_in:
            with gzip.open(f'{logfile}.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        logfile.unlink()

    _ch = _log.StreamHandler(open(logfile, 'a'))
    _formatter = _log.Formatter('%(name)s (%(module)s.%(funcName)s) %(levelname)s: %(message)s')
    _ch.setFormatter(_formatter)
    pa_logger.addHandler(_ch)



_log.basicConfig(level=_log.INFO)
io_logger = _log.getLogger('procastro.io')
io_logger.propagate = False

