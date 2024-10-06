import numpy as np
import json
import random


def compute_legendre(data: dict):

    data["legendre"] = dict()
    data["legendre"]["description"] = "Collection of Legendre quadrature points and weights"
    data["legendre"]["values"] = dict()


    n = [2,3,4,5,6,11,26,37]

    for ni in n:
        leg_file = 'quadrature-data/Le%i.txt' % ni
        leg_data = np.loadtxt(leg_file)
        points = leg_data[:,0]
        weights = leg_data[:,1]
        data["legendre"]["values"]["n{}".format(ni)] = {"points": points.tolist(), "weights": weights.tolist()}



def compute_lobatto(data: dict):
    data["lobatto"] = dict()    
    data["lobatto"]["description"] = "Collection of Lobatto quadrature points and weights"
    data["lobatto"]["values"] = dict()

    n = [2,3,4,5,6,11,26,37]

    for ni in n:
        lob_file = 'quadrature-data/Lo%i.txt' % ni
        lob_data = np.loadtxt(lob_file)
        points = lob_data[:,0]
        weights = lob_data[:,1]
        data["lobatto"]["values"]["n{}".format(ni)] = {"points":points.tolist(), "weights":weights.tolist()}

    




def main():

    
    data1 = dict()
    compute_legendre(data1)
    compute_lobatto(data1)
        

    with open('gauss-quad.json', 'w') as f:
        json.dump(data1, f, indent=4)






if __name__ == '__main__':
    main()
