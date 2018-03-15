
import numpy as np


''' iterate by checking with epislon or by epoch=100, then do sigmoid '''

''' Constants '''
N = 10000



def gradient(y, x, w, x_test, y_test):

    stepSize = 0.0001
    print("The Step Size is " + str(stepSize) + ". Following are the values for each iteration:")
    for epoch in range(300):
        grad_sum = 0
        objectiveFuncSum = 0

        for n in range(0, N):

            power = y[n] * np.dot(np.transpose(w), x[n])
            grad_sum = grad_sum + (y[n]*x[n] / (1 + np.exp( power )))

            ''' compute objective function  and show that it is decreasing '''
            objectiveFuncSum += np.log( 1 + np.exp(-1*y[n] * np.dot(np.transpose(w), x[n])) )

        w = w - stepSize*(-1*(1/N))*grad_sum
        accuracy = test_weights(x_test, y_test, w)
        gradient = np.linalg.norm(grad_sum)

        print( str(epoch+1) + "\t" + str(objectiveFuncSum/N) + "\t" +  str(gradient) + "\t" + str(accuracy))
    return w


def test_weights(x_test, y_test, trained_w):

    n_correct_weights = 0

    for i in range( len(x_test[0])):

        product = np.dot( np.transpose(trained_w), x_test[i])

        if product >= 0:
            product = 1

        elif product < 0:
            product = -1



        if product == y_test[i]:

            n_correct_weights += 1

    accuracy = n_correct_weights/ len(x_test[0])
    return accuracy

def main():
    x = np.loadtxt("mnist_2_vs_7/mnist_X_train.dat")
    y = np.loadtxt("mnist_2_vs_7/mnist_y_train.dat")
    x_test = np.loadtxt("mnist_2_vs_7/mnist_X_test.dat")
    y_test = np.loadtxt("mnist_2_vs_7/mnist_y_test.dat")

    w = np.random.uniform(0, 0.01, size=780)
    #N = np.shape(X)[0]
    trained_w = gradient(y, x , w, x_test, y_test)


if __name__ == "__main__":
    main()
