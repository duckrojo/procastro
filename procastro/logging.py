import logging as _log

# Eliminar la configuración del logger raíz (ELIMINA ESTA LÍNEA)
# _log.basicConfig(level=_log.INFO)

# Configurar el logger de procastro para manejar todos sus mensajes
dplogger = _log.getLogger('procastro')
dplogger.setLevel(_log.INFO)
dplogger.propagate = False  # Evita que los mensajes se propaguen al logger raíz

# Configuración del handler para procastro
_ch = _log.StreamHandler()
_formatter = _log.Formatter('%(name)s (%(module)s.%(funcName)s) %(levelname)s: %(message)s')
_ch.setFormatter(_formatter)
dplogger.addHandler(_ch)

# Mantén la configuración existente para io_logger
io_logger = _log.getLogger('procastro.io')
io_logger.propagate = False