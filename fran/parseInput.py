__author__ = 'fran'

""" Esto es solo para parsear el input inicial que estaria dado por la GUI.
Despues se pueden agregar mas archivos y etc a lo que salga de aca. """

from dataproc.core import io_dir, io_file

def parse(*args, **kwargs):
    all_input = {'bias': None, 'dark': None, 'flat': None, 'sci': None}
    for kind, path in kwargs.items():
        all_input[kind] = io_dir.AstroDir(path)

    return all_input

