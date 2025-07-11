from procastro.axis.axis import AstroAxis


class ChannelAxis(AstroAxis):
    acronym = "C"
    selectable = True
    discrete = True

    def __init__(self,
                 nn,
                 values: list[str],
                 ):
        super().__init__(len(values),
                         values = values,
                         )

    def short(self):
        return "Information Channels"