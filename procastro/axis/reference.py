from procastro.axis.axis import AstroAxis


class ReferenceAxis(AstroAxis):
    acronym = "R"
    discrete = True

    def short(self):
        return f"Reference"