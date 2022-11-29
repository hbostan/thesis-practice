import math


class Vec2:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        # self.slope = y/x
        self.length = math.sqrt(self.x**2 + self.y**2)

    def __repr__(self):
        return f'Vec2({self.x:.4f},{self.y:.4f})'

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __mul__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x * other.x, self.y * other.y)
        return Vec2(self.x * other, self.y * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, divisor):
        return Vec2(self.x / divisor, self.y / divisor)

    def __floordiv__(self, divisor):
        return Vec2(self.x // divisor, self.y // divisor)

    def __eq__(self, other):
        if other != None and self.x == other.x and self.y == other.y:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def distance_to(self, p):
        ds = (p.x - self.x)**2 + (p.y - self.y)**2
        return math.sqrt(ds)

    def normal(self):
        return Vec2(self.y, -self.x)

    def normalize(self):
        return self / self.length

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y)
