
import numpy as np


''' iterate by checking with epislon or by epoch=100, then do sigmoid '''

''' Constants '''
N = 10000

# Function Taken from  https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison as posted by the TA
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def stochasticGradient( y_unshuffled, x_unshuffled, w, x_test, y_test):

    stepSize = 0.0001
    index = 0

    batchSize = 100
    x_shuffled,y_shuffled = unison_shuffled_copies(x_unshuffled, y_unshuffled)

    for i in range(300):

        x = x_shuffled[ index*batchSize : (index+1)*batchSize ]
        y = y_shuffled[ index*batchSize : (index+1)*batchSize ]


        if( (index+1)*batchSize >= N ):
            '''shuffle batches'''
            index = -1
            x_shuffled,y_shuffled = unison_shuffled_copies(x_unshuffled, y_unshuffled)



        objectiveFuncSum = 0
        stochastic_sum = 0
        for n in range(batchSize):
            exponent = np.exp( y[n]*( np.dot( np.transpose(w), x[n])))
            stochastic_sum +=  y[n] * x[n] / ( 1 + exponent)
            objectiveFuncSum += np.log( 1 + np.exp(-1*y[n] * np.dot(np.transpose(w), x[n])) )

        if (i+1)%50 == 0:
            stepSize = stepSize * 0.5

        w = w - stepSize * (-1 * stochastic_sum / batchSize)

        accuracy = test_weights(x_test, y_test, w)
        gradient = np.linalg.norm(stochastic_sum)

        #print("Epoch: " + str(i+1) + "\t" "Objective Function: " + str(objectiveFuncSum/batchSize) + "\t" + "Gradient: " + str(gradient) + "\t" + "Accuracy: " + str(accuracy))
        print(str(i+1) +"\t" + str(objectiveFuncSum/batchSize) + "\t" + str(gradient) + "\t"  + str(accuracy))

        index += 1





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

    stochasticGradient(y, x , w, x_test, y_test)

if __name__ == "__main__":
    main()
