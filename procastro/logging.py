

import logging as _log

_log.basicConfig(level=_log.INFO)
io_logger = _log.getLogger('procastro.io')
io_logger.propagate = False

dplogger = _log.getLogger('procastro')
_ch = _log.StreamHandler()
_formatter = _log.Formatter('%(name)s (%(module)s.%(funcName)s) %(levelname)s: %(message)s')
_ch.setFormatter(_formatter)
dplogger.addHandler(_ch)
