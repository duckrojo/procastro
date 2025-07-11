from unittest import TestCase

from procastro.data.utils import DictInitials


class test_utils(TestCase):

    def test_correct_fails(self):
        dct = DictInitials(key1=1, key2=2, other=3, here=4, key=5, )
        with self.assertRaises(IndexError):
            test = dct['ke']
        with self.assertRaises(IndexError):
            test = dct['none']

    def test_correct_returns(self):
        dct = DictInitials(key1=1, key2=2, other=3, here=4, key=5, )

        assert dct['key1'] == 1
        assert dct['o'] == 3
        assert dct['key'] == 5

