import re
import warnings

from procastro import config
from procastro.misc.general import accept_object_name

import pyvo as vo
exo_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")


def get_transit_ephemeris_file(target):
    """
    Recovers epoch, period and length of a target transit if said transit has
    been specified in one of the provided paths

    Transit files must be named ".transits" or "transits.txt", each transit
    should have the following columns separated by a space:

    {object_name} E{transit_epoch} P{transit_period} L{transit_length}

    If the object name contains spaces, replace them with an underscore when
    writing it into the file. On the other hand, querying a name with spaces
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

    config_exo = config.config_user("exoplanet")
    paths = [config_exo['transit_file'],
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

                if accept_object_name(planet, target, planet_match=True):
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

def query_transit_ephemeris(target):

    if target[-1] not in 'bcdefghij':
        target_fmtd = re.sub(r"(k2|[a-zA-Z]+)-?(\d+)([A-D]?) ?",
                         r"\1-\2 b",
                             target
                             ).lower()
    else:
        target_fmtd = re.sub(r"(k2|[a-zA-Z]+)-?(\d+)([A-D]?) ?([b-g]?)",
                             r"\1-\2 \3",
                             target
                             ).lower()

    print(f"Attempting to query transit information for '{target}' -> '{target_fmtd}'")

    query = f"SELECT pl_name,pl_tranmid,pl_orbper,pl_trandur FROM exo_tap.pscomppars " \
            f"WHERE lower(pl_name) like '%{target_fmtd}%' "
    resultset = exo_service.search(query)
    try:
        req_cols = [resultset['pl_orbper'].data[0], resultset['pl_tranmid'].data[0]]
    except IndexError:
        raise IndexError(f"Planet {target_fmtd} not found in exoplanet database")
    trandur = resultset['pl_trandur'].data[0]
    if trandur is None:
        req_cols.append(1)
        warnings.warn("Using default 1hr length for transit duration", UserWarning)
    else:
        req_cols.append(trandur)

    transit_period, transit_epoch, transit_length = req_cols

    print("  Found ephemeris: {0:f} + E*{1:f} (length: {2:f})"
          .format(transit_epoch, transit_period, transit_length))

    return transit_epoch, transit_period, transit_length
