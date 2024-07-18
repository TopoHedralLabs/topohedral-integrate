import sympy as sy


def test1():

    print('..............................')
    x = sy.Symbol('x')
    y = sy.Symbol('y')

    f = 0.3 * x**4 * y**4 + 2 * x**3 * y**3 - 0.1 * x**2 * y**2 + 100*x*y + 200
    integral_f = sy.integrate(f, (x, -0.3, 5), (y, -3, 2))
    print(integral_f)

def test2():

    print('..............................')
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    f = sy.sin(x) * sy.sin(y)
    integral_f = sy.integrate(f, (x, 0, 30), (y, 0, 30))
    sy.pprint(integral_f)

def test3():

    print('..............................')
    x = sy.Symbol('x')
    y = sy.Symbol('y')
    f = sy.Abs(x + 1) * sy.Abs(y - 2)
    integral_f = sy.integrate(f, (x, -3, 4), (y, 0, 5))
    sy.pprint(integral_f)


def main():

    test1()
    test2()
    test3()


if __name__ == '__main__':
    main()