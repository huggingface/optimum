import unittest
from unittest import TestCase

class TestFun(TestCase):
    def test_is_string(self):
        s = "test"
        self.assertTrue(isinstance(s, basestring))

if __name__ == '__main__':
    unittest.main()