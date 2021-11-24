#
# dataproc - general data processing routines
#
# Copyright (C) 2021 Patricio Rojo
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of version 2 of the GNU General
# Public License as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
#

import logging

iologger = logging.getLogger('dataproc.io')


class AstroCommon:
    @staticmethod
    def parse_filter(**kwargs):
        def cast_identity_val(val, req):
            return val

        def cast_identity_req(val, req):
            return req

        def cast_icase_val(val, req):
            return val.lower()

        def cast_icase_req(val, req):
            return req.lower()

        def cast_begin(val, req):
            return val[:len(req)]

        def cast_end(val, req):
            return val[-len(req):]

        def op_not(val, req):
            return req != val

        def op_match(val, req):
            return req in val

        def op_equal(val, req):
            return req == val

        def op_lt(val, req):
            return val < req

        def op_gt(val, req):
            return val > req

        filters = []

        for filter_keyword, request in kwargs.items():
            functions = []
            # By default is not comparing match, but rather equality
            match = False
            exists = True

            def cast(x):
                return x

            filter_keyword = filter_keyword.replace('__', '-')
            if '_' in filter_keyword:
                tmp = filter_keyword.split('_')
                filter_keyword = tmp[0]
                functions.extend(tmp[1:])

            if isinstance(request, str):
                request = [request]
            elif isinstance(request, (tuple, list)):
                raise TypeError("Filter string cannot be tuple/list anymore. "
                                "It has to be a dictionary with the casting "
                                "function as key (e.g. {'str': ['5', '4']})")
            elif isinstance(request, dict):
                keys = list(request.keys())
                if len(keys) != 1:
                    raise NotImplementedError(
                        "Only a single key (casting) per filtering function "
                        "has been implemented for multiple alternatives")
                try:
                    request = list(request[keys[0]])
                    if 'begin' in functions or 'end' in functions:
                        raise ValueError("Cannot use '_begin' or '_end'"
                                         "if comparing to a list")
                except TypeError:
                    request = [request[keys[0]]]
                cast = eval(keys[0])
                if not callable(cast):
                    raise ValueError(
                        "Dictionary key (casting) has to be a callable "
                        "function accepting only one argument")
            else:
                cast = type(request)
                request = [request]

            value_cast = cast_identity_val
            request_cast = cast_identity_req
            operation = False
            for f in functions:
                f = f.lower()
                if f == 'begin':
                    value_cast = cast_begin
                elif f == 'end':
                    value_cast = cast_end
                elif f == 'icase':
                    value_cast = cast_icase_val
                    request_cast = cast_icase_req
                elif f"op_{f}" in locals():
                    operation = eval(f"op_{f}")
                else:
                    iologger.warning(f"Function '{f}' not recognized in "
                                     f"filtering, ignoring")

            filters.append([value_cast, request_cast, operation, request, filter_keyword])

        return filters
