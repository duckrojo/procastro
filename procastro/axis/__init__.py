import importlib
from os import listdir
from os.path import dirname

modules = [name[:-3] for name in listdir(dirname(__file__)) if name[0] != '_']

for module in modules:
    imported = importlib.import_module("."+module, __name__)
    current_globals = globals()

    if hasattr(imported, '__all__'):
        for name in imported.__all__:
            current_globals[name] = getattr(imported, name)
    else:
        for name in dir(imported):
            if not name.startswith('_') and name not in current_globals:
                current_globals[name] = getattr(imported, name)

    del current_globals[module]

del imported
del modules
del module
del current_globals
del importlib
del name
del dirname
del listdir

