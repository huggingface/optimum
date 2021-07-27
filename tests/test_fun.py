import unittest
from unittest import TestCase


class TestFun(TestCase):
    def test_is_string(self):
        s = 1
        self.assertEqual(s, 1)


if __name__ == "__main__":
    unittest.main()
