import unittest

from PIL import Image

from src.data import tw_face


class TestData(unittest.TestCase):
    def setUp(self):
        self.old_size = 293
        self.young_size = 334

    def test_old_count(self):
        x, y = tw_face.read_data('old')
        self.assertEqual(len(x), self.old_size)
        self.assertEqual(y.shape, (self.old_size, ))

    def test_young_count(self):
        x, y = tw_face.read_data('young')
        self.assertEqual(len(x), self.young_size)
        self.assertEqual(y.shape, (self.young_size, ))

    def test_all_count(self):
        x, y = tw_face.read_data()
        self.assertEqual(len(x), self.young_size + self.old_size)
        self.assertEqual(y.shape, (self.young_size + self.old_size, ))

    def test_readable(self):
        x, y = tw_face.read_data()
        for xx in x:
            Image.open(xx).close()

    def test_label(self):
        x, y = tw_face.read_data()
        for yy in y.flat:        
            self.assertIn(yy, range(7))
