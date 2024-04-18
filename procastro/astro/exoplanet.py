import procastro as pa


def get_transit_ephemeris(target):
    """
    Recovers epoch, period and length of a target transit if said transit has
    been specified in one of the provided paths

    Transit files must be named ".transits" or "transits.txt", each transit
    should have the following columns separated by a space:

    {object_name} E{transit_epoch} P{transit_period} L{transit_length}

    If the object name contain spaces, replace them with an underscore when
    writing it into the file. On the other hand querying a name with spaces
    requires using spaces.

    An optional comment column can be placed at the end of a row placing a-mass
    'C' as prefix.

    Parameters
    ----------
    target: str
        Target requested

    Returns
    -------
    tr_epoch : float or None
    tr_period : float or None
    tr_length : float or None

    Raises
    ------
    ValueError
        If a data field does not match the specified format
    """
    paths = [pa.user_confdir("transits.txt")
             ]

    tr_epoch = None
    tr_period = None
    tr_length = None
    for transit_filename in paths:
        try:
            open_file = open(transit_filename)

            override = []
            print(transit_filename)
            for line in open_file.readlines():
                if line[0] == '#' or len(line) < 3:
                    continue
                data = line[:-1].split()
                planet = data.pop(0)

                if pa.accept_object_name(planet, target, planet_match=True):
                    for d in data:
                        if d[0].lower() == 'p':
                            override.append('period')
                            tr_period = float(eval(d[1:]))
                        elif d[0].lower() == 'e':
                            override.append("epoch")
                            tr_epoch = float(eval(d[1:]))
                        elif d[0].lower() == 'l':
                            override.append("length")
                            tr_length = float(eval(d[1:]))
                        elif d[0].lower() == 'c':
                            override.append("comment")
                        else:
                            raise ValueError("data field not understood, it "
                                             "must start with L, P, C, "
                                             "or E:\n{0:s}".format(line,))
                    print(f"Overriding for '{planet}' from file '{transit_filename}': {', '.join(override)}")

                if len(override):
                    break

        except IOError:
            pass

    return tr_epoch, tr_period, tr_length
