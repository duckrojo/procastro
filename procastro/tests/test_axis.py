import re
import procastro.axis as paa

def test_list_acronyms_is_unique():
    axis = paa.AstroAxis()
    acronyms = axis.list_acronyms()
    print(f"Found the following types of AstroAxis: {acronyms}")

    assert len(acronyms) == len(paa.AstroAxis._available_axis) == len(set(acronyms))


def test_acronyms_have_format():
    axis = paa.AstroAxis()
    acronyms = axis.list_acronyms()
    print(f"Found the following types of AstroAxis: {acronyms}")

    for acronym in acronyms:
        assert re.match("[A-Z][a-z]?", acronym)


def test_use_chooses_correct_child():
    axis = paa.AstroAxis()
    acronyms = axis.list_acronyms()
    print(f"Found the following types of AstroAxis: {acronyms}")

    for acronym in acronyms:
        assert axis.use(acronym).acronym == acronym
