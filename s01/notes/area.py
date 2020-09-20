import numpy as np

x = np.arange(0, 1, 0.001)
y = np.sqrt(1 - x ** 2)

points = [(0, 0), (0, 2), (2, 2), ]
x, y = zip(*points)


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calc_area(bbox):
    x = [p for i, p in enumerate(bbox) if (i % 2) == 0]
    y = [p for i, p in enumerate(bbox) if (i % 2) == 1]
    return poly_area(x, y)


print(poly_area(x, y))
print(calc_area([p for sublist in points for p in sublist]))
