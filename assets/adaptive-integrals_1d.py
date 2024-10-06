
import sympy as sy


def test1():

    print('...................................')
    x = sy.Symbol('x')
    f = 7 * x**4 + 2 * x**3 - 11 * x**2  + 15*x + 1
    integral_f = sy.integrate(f, (x, -3, 10))
    print(integral_f)   


def test2():

    print('...................................')
    x = sy.Symbol('x')
    f = sy.sin(x)
    integral_f = sy.integrate(f, (x, 0, 30))
    print(integral_f)   

def test3():

    print('...................................')
    x = sy.Symbol('x')
    f = sy.Abs(x + 1)
    integral_f = sy.integrate(f, (x, -3, 4))

    print(integral_f)   


def test4():

    print('...................................')
    x = sy.Symbol('x')
    f = sy.exp(-x**2)
    integral_f = sy.integrate(f, (x, -3, 3))
    print(integral_f)
    print(integral_f.evalf())   

def test5():

    print('...................................')
    x = sy.Symbol('x')
    f = sy.ln(x)
    F = sy.integrate(f, x)  
    integral_f = sy.integrate(f, (x, 0, 10))
    p = sy.plot(f, (x, 0, 10), show=False)  
    p.show()
    sy.pprint(F)
    print(integral_f)

def main():
    test1()
    test2()
    test3()
    test4()
    test5()



if __name__ == '__main__':
    main()  