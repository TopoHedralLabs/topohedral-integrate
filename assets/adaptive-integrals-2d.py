import sympy as sy


def test1():

    print('..............................')
    x = sy.Symbol('x')
    y = sy.Symbol('y')

    f = 0.3 * x**4 * y**4 + 2 * x**3 * y**3 - 0.1 * x**2 * y**2 + 100*x*y + 200
    integral_f = sy.integrate(f, (x, -0.3, 5), (y, -3, 2))
    print(integral_f)


def main():

    test1()


if __name__ == '__main__':
    main()