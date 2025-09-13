from pathlib import Path
import sys
import unittest

import numpy as np

f = Path(__file__)
s = f.parent.parent / "src"
sys.path.append(str(s))
print(__file__)
del f, s
import bezier  # type: ignore


class TestLinear(unittest.TestCase):
    def testPointsShape(self) -> None:
        N = 100
        start = [1.0, 0.0, 0.0]
        stop = [0.0, 1.0, 0.0]
        t = np.linspace(0.0, 1.0, N + 1, endpoint=True, retstep=False)
        self.assertEqual(t.shape, (N + 1,))
        x0 = np.array([start, stop])
        self.assertEqual(x0.shape, (2, 3))
        x = bezier.line(t, x0)
        self.assertEqual(x.shape, (3, N + 1))

    def testPointsValues(self) -> None:
        N = 100
        start = [1.0, 0.0, 0.0]
        stop = [0.0, 1.0, 0.0]
        t = np.linspace(0.0, 1.0, N + 1, endpoint=True, retstep=False)
        self.assertEqual(t.shape, (N + 1,))
        x0 = np.array([start, stop])
        x = bezier.line(t, x0)
        for index in range(3):
            self.assertEqual(x[index, 0], start[index])
            self.assertEqual(x[index, N], stop[index])
        self.assertAlmostEqual(x[0, N // 2], 0.5)
        self.assertAlmostEqual(x[1, N // 2], 0.5)


class TestCross(unittest.TestCase):
    def testShape(self) -> None:
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        zz = bezier.cross(x, y)
        self.assertEqual(zz.shape, (3,))

    def testXYZ(self) -> None:
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        zz = bezier.cross(x, y)
        self.assertAlmostEqual(zz[0], z[0])
        self.assertAlmostEqual(zz[1], z[1])
        self.assertAlmostEqual(zz[2], z[2])

    def testParallel(self):
        x = np.array([1.0, 0.0, 0.0])
        z = bezier.cross(x, x)
        self.assertAlmostEqual(z[0], 0.0)
        self.assertAlmostEqual(z[1], 0.0)
        self.assertAlmostEqual(z[2], 0.0)


class TestProcess(unittest.TestCase):
    def testValues1(self):
        x = [1.0, 0.0, 0.0]
        y = [0.0, 1.0, 0.0]
        xx, yy, mm, nn, a, b, c, theta = bezier.process(x, y)
        # xx
        self.assertAlmostEqual(xx[0], 1.0)
        self.assertAlmostEqual(xx[1], 0.0)
        self.assertAlmostEqual(xx[2], 0.0)
        # yy
        self.assertAlmostEqual(yy[0], 0.0)
        self.assertAlmostEqual(yy[1], 1.0)
        self.assertAlmostEqual(yy[2], 0.0)
        # mm
        self.assertAlmostEqual(mm[0], 0.0)
        self.assertAlmostEqual(mm[1], 0.0)
        self.assertAlmostEqual(mm[2], 1.0)
        # nn
        self.assertAlmostEqual(nn[0], 0.0)
        self.assertAlmostEqual(nn[1], 1.0)
        self.assertAlmostEqual(nn[2], 0.0)
        # coordinates
        self.assertAlmostEqual(a, 1.0)
        self.assertAlmostEqual(b, 0.0)
        self.assertAlmostEqual(c, 1.0)
        # theta
        self.assertAlmostEqual(theta, np.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
