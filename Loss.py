
import numpy as np
import matplotlib.pyplot as plt

def L1_Loss(x):
    return np.abs(x)

def L2_Loss(x):
    return (x ** 2)

def Smooth_L1_Loss(x):
    '''
    smooth_l1_loss = {
        0.5 * (x ** 2), if |x| < 1
        |x| - 0.5     , otherwise
    }
    '''
    x = np.abs(x)
    smooth_l1_loss = np.where(x < 1.0, 0.5 * (x ** 2), x - 0.5)

    return smooth_l1_loss

def Smooth_L1_Loss_with_Sigma(x, sigma = 1.5):
    '''
    smooth_l1_loss = {
        0.5 * (x ** 2), if |x| < 1
        |x| - 0.5     , otherwise
    }
    '''
    sigma2 = sigma * sigma

    x = np.abs(x)
    smooth_l1_loss = np.where(x < (1.0 / sigma2), sigma2 * 0.5 * (x ** 2), x - 0.5 / sigma2)
    return smooth_l1_loss

if __name__ == '__main__':

    x_list = []
    l1_list = []
    l2_list = []
    smooth_l1_list = []
    smooth_l1_sigma_1_list = []
    smooth_l1_sigma_2_list = []
    
    x = -6.
    alpha = 0.01

    while x <= 6.:
        x_list.append(x)
        
        l1_list.append(L1_Loss(x))
        l2_list.append(L2_Loss(x))
        smooth_l1_list.append(Smooth_L1_Loss(x))
        smooth_l1_sigma_1_list.append(Smooth_L1_Loss_with_Sigma(x, sigma = 1.5))
        smooth_l1_sigma_2_list.append(Smooth_L1_Loss_with_Sigma(x, sigma = 2.5))

        x += alpha

    plt.plot(x_list, smooth_l1_list, 'red')
    plt.plot(x_list, smooth_l1_sigma_1_list, 'orange')
    plt.plot(x_list, smooth_l1_sigma_2_list, 'green')
    plt.plot(x_list, l1_list, 'blue')
    plt.plot(x_list, l2_list, 'purple')

    plt.xlim(-3, 3)
    plt.ylim(0, 3)
    plt.legend(['Smooth_L1', 'Smooth_L1_with_Sigma_1', 'Smooth_L1_with_Sigma_2', 'L1', 'L2'], loc = 'upper center')

    plt.show()

