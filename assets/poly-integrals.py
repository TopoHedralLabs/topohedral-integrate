
import random
import json


def compute_integrals_1d(data: dict):

    data["description"] = "Analytic integrals of polynomials in 1D"
    data["values"] = dict()

    a = 3.4
    b = 9.7
    data["values"]["range"] = [a, b]

    for i in range(16):

        coeffs = [random.uniform(-10.0, 10.0) for j in range(i+1)]
        integral = 0.0
        for j in range(i+1):
            integral += (coeffs[j] / (j + 1)) * (b**(j+1) - a**(j+1))

        data["values"]["P{}".format(i)] = {"coeffs": coeffs, "integral": integral}

def compute_integrals_2d(data: dict):

    data["description"] = "Analytic integrals of polynomials in 2D"
    data["values"] = dict()

    a = 3.4
    b = 9.7
    c = -3.5
    d = 2.1

    data["values"]["range"] = [a, b, c, d]

    for i in range(16):

        coeffs_u = [random.uniform(-10.0, 10.0) for j in range(i+1)]
        integral_u = 0.0
        for j in range(i+1):
            integral_u += (coeffs_u[j] / (j + 1)) * (b**(j+1) - a**(j+1))

        
        coeffs_v = [random.uniform(-10.0, 10.0) for j in range(i+1)]
        integral_v = 0.0
        for j in range(i+1):
            integral_v += (coeffs_v[j] / (j + 1)) * (d**(j+1) - c**(j+1))

        integral = integral_u * integral_v

        data["values"]["P{}".format(i)] = {"coeffs_u": coeffs_u, "coeffs_v": coeffs_v, "integral": integral}

def main():

    data_1d = dict()
    compute_integrals_1d(data_1d)
    with open('poly-integrals-1d.json', 'w') as f:
        json.dump(data_1d, f, indent=4)


    data_2d = dict()
    compute_integrals_2d(data_2d)
    with open('poly-integrals-2d.json', 'w') as f:
        json.dump(data_2d, f, indent=4)
    


if __name__ == '__main__':
    main()