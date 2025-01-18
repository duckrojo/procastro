
class CalibBase:
    def __init__(self, **kwargs):
        self.has_bias = self.has_flat = False

        self.bias: 'AstroFileBase | float' = 0.0
        self.flat: 'AstroFileBase | float' = 1.0

    def short(self):
        return f"({'B' if self.has_bias else ''}{'F' if self.has_flat else ''})"

    def __str__(self):
        ret = []
        for label in ['bias', 'flat']:
            if hasattr(self, f"has_{label}"):
                ret.append(label)
        return f"Calib: {' + '.join(ret)}"
