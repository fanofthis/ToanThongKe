# Import libary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from __future__ import division, print_function, unicode_literals

def main():
    # path store dataset
    filePath = "Dataset\\01_MHEALTH.xls"

    # read data
    df = pd.read_excel(filePath, index_col = None)

    # Height (inch)
    height = np.array([list(df.HT)]).T

    # Weight (lb)
    weight = np.array(list([df.WT])).T

    # Visualize data
    plt.plot(height, weight, 'ro')

    #plt.axis([50, 80, 120, 250])
    plt.xlabel('Height (inch)')
    plt.ylabel('Weight (lb)')
    plt.show()

    # Calculate weight of fitting line
    one = np.ones((height.shape[0],1))
    Xbar = np.concatenate((one, height), axis = 1)

    A = np.dot(Xbar.T, Xbar)
    b = np.dot(Xbar.T, weight)
    w = np.dot(np.linalg.pinv(A), b)
    #print(w)

    # Calculate the correlation coefficient
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(62,80,2)
    y0 = w_0 + w_1*x0

    # Drawing line fitting line
    plt.plot(height.T, weight.T, 'ro')
    plt.plot(x0,y0)
    plt.xlabel('Height (inch)')
    plt.ylabel('Weight (lb)')
    plt.show()

    # Preduct weight base height
    # Set hight to predict weight
    height_P = 72

    # Linear regression
    weight_P = w_1 * height_P + w_0
    print("Phuong Trinh Hoi Quy la: {} * height + {}".format(round(w_1,2),round(w_0,2)))
    print("Weight predict is {} lb if height is {} inch .".format(weight_P, height_P))


if __name__ == "__main__":
    main()