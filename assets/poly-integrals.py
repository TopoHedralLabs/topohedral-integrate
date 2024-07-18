
import random
import json


def compute_integrals(data: dict):

    data["description"] = "Analytic integrals of polynomials"
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

def main():

    data2 = dict()
    compute_integrals(data2)
    with open('poly-integrals.json', 'w') as f:
        json.dump(data2, f, indent=4)


if __name__ == '__main__':
    main()