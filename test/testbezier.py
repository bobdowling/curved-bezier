from pathlib import Path
import sys
import unittest

import numpy

f = Path(__file__)
s = f.parent.parent / "src"
sys.path.append(str(s))
print(__file__)
del f, s
import bezier  # type: ignore


class TestLinear(unittest.TestCase):
    def testPoints(self) -> None:
        N = 100
        start = [1.0, 0.0, 0.0]
        stop = [0.0, 1.0, 0.0]
        t = numpy.linspace(0.0, 1.0, N + 1, endpoint=True, retstep=False)
        self.assertEqual(t.shape, (N + 1,))
        x0 = numpy.array([start, stop])
        self.assertEqual(x0.shape, (2, 3))
        x = bezier.line(t, x0)
        self.assertEqual(x.shape, (3, N + 1))
        for index in range(3):
            self.assertEqual(x[index, 0], start[index])
            self.assertEqual(x[index, N], stop[index])


if __name__ == "__main__":
    unittest.main()
