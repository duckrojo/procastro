__author__ = 'fran'

import subject as so
from clint.textui import prompt, validators, colored, indent, puts


def main():
    #print('Welcome to ADO')

    # Create a sharedObject -- whatever it's going to be
    obj = so.Subject(None, None)  # TODO poder inicializar sharedObject vacio
    eye = so.Eye(None, {}, [])
    #eye.listMethods()
    #toopen = prompt.query('Installation Path', default='/usr/local/bin/', validators=[validators.PathValidator()])

    with indent(1, quote=colored.red('> ')):
        puts('Welcome!')
        puts('Please enter usage path')

    toopen = prompt.query('Path', default='/usr/local/bin/', validators=[validators.PathValidator()])
    print(toopen)

    with indent(1, quote=colored.red('> ')):
        puts('OPTIONS:')
        with indent(1, quote=colored.yellow('> ')):
            puts('name = observable: create empty AstroFile Observable')
            puts('open filename: create AstroFile Observable with data filename')
            puts('name = observer: create Observer')
            puts('attach observablename observername: attach observer to observable')
            puts('methods: list all methods')
            puts('methods ini: list all methods starting with ini')
            puts('methodname(observable): apply method to observable')

    while True:
        try:
            inst = prompt.query('>>')

            if inst == 'q':
                break

            if inst[-10:] == 'observable':
                obsname = inst.split('=')[0]
                if obsname[0][-1:] == ' ':
                    obsname = obsname[0][:-1]
                print(obsname)
        except (KeyboardInterrupt, SystemExit):
            break

    print('bye!')




if __name__ == '__main__':
    main()
