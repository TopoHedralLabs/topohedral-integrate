
import sympy as sy


def test1():

    x = sy.Symbol('x')
    f = 7 * x**4 + 2 * x**3 - 11 * x**2  + 15*x + 1
    integral_f = sy.integrate(f, (x, -3, 10))
    print(integral_f)   


def test2():

    x = sy.Symbol('x')
    f = sy.sin(x)
    integral_f = sy.integrate(f, (x, 0, 30))
    print(integral_f)   

def main():
    test1()
    test2()



if __name__ == '__main__':
    main()  