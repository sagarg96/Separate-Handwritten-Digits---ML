
import numpy as np

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

    for i in range(0,300):

        x = x_shuffled[ index*batchSize : (index+1)*batchSize ]
        y = y_shuffled[ index*batchSize : (index+1)*batchSize ]


        if( (index+1)*batchSize >= N ):
            '''shuffle batches'''
            index = -1
            x_shuffled,y_shuffled = unison_shuffled_copies(x_unshuffled, y_unshuffled)

        objectiveFuncSum = 0
        stochastic_sum = 0
        for n in range(batchSize):

            ywn = y[n] * np.dot( np.transpose(w), x[n])
            stochastic_sum = stochastic_sum + ((-1* y[n]*x[n]) if ywn < 1 else 0)
            objectiveFuncSum += max(0, 1-ywn)

        if (i+1)%50 == 0:
            stepSize = stepSize*0.5


        w = w - stepSize * ( stochastic_sum / batchSize)

        accuracy = test_weights(x_test, y_test, w)
        gradient = np.linalg.norm(stochastic_sum)

        print("Epoch: " + str(i+1) + "\t" "Objective Function: " + str(objectiveFuncSum/batchSize) + "\t" + "Gradient: " + str(gradient) + "\t" + "Accuracy: " + str(accuracy))


        index += 1


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
    trained_w = stochasticGradient(y, x , w, x_test, y_test)
    # trained_w = gradient(y, x, w)
    test_weights(x_test, y_test, trained_w)

    #stochasticGradient(y, x, w)

if __name__ == "__main__":
    main()
