#This code uses a neural network and the Metropolis Algorithm to produce a truth table for a 3-input XOR logic gate
import matplotlib.pyplot as plt
import numpy as np

# randomise all my weights uniformly between -1 and 1
w = 2.0*np.random.random(21)-1.0
# define my parameters, including stepsize and temprature for the metropolis algorithm
k = 1
stepsize = 1
temp = 0.001
correct = 0
ite = 0
# sets the output of the neural network as an array for all 8 outputs
o8 = np.zeros(8)
# sets an array which records the error value after every iteration of the metropolis algorithm
error = []

# truth table for 3-input XOR logic gate
t = [[0, 0, 0, 0],
     [0, 0, 1, 1],
     [0, 1, 0, 1],
     [0, 1, 1, 0],
     [1, 0, 0, 1],
     [1, 0, 1, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 1]]

# sigmoid threshold function


def stf(x):
    return 1.0/(1+np.exp(-k*x))  # maybe change to np.exp #CHECKKKKKK

# function which does the neural network calculations and outputs how correct it is and the error


def ntest():
    good = 0
    err = 0.0
    for i in range(8):
        i1 = t[i][0]
        i2 = t[i][1]
        i3 = t[i][2]  # tick
        h4 = stf(w[0]+w[5]*i1+w[6]*i2+w[7]*i3)  # tick
        h5 = stf(w[1]+w[8]*i1+w[9]*i2+w[10]*i3)  # tick
        h6 = stf(w[2]+w[11]*i1+w[12]*i2+w[13]*i3)  # tick
        h7 = stf(w[3]+w[14]*i1+w[15]*i2+w[16]*i3)  # tick
        o8[i] = stf(w[4]+w[17]*h4+w[18]*h5+w[19]*h6+w[20]*h7)  # tick
        err = err+0.5*(o8[i]-t[i][3])**2
        if(int(round(o8[i]) == t[i][3])):
            good += 1
    return good, err, o8


# loop metropolis over maximum 5000 iterations, if it doesnt get it right in that time itll output the incorrect results
for ite in range(0, 5000):
    # compute neural netweork calculations for current weights
    correct, err_0, out = ntest()
    # test if correct output
    if correct == 8:
        break
    # make a copy of the current weights, in case the proposed weight change in metropolis algorithm is rejected
    w_og = np.copy(w)
    # metropolis algorithm loop over all weights
    for i in range(21):
        # randomly generate a variable num1 between -1 and 1
        num1 = np.random.uniform(-1.0, 1.0)
        # propose weight change
        w[i] += num1*stepsize
        # compute new error from single updated weight
        good, new_err, out = ntest()
        # calculate the change in error
        delta_err = new_err-err_0
        # if the change in error is negative (the new error is lower), accept the weight change and set the new error as the error
        if delta_err < 0:
            err_0 = new_err
        # otherwise choose a random number, p, in the range (0,1)
        else:
            num2 = np.random.uniform(0.0, 1.0)
            # and if this num2 value is less than the exp(-delta_err/temp), then accept the weight change
            if num2 < np.exp((-1*delta_err)/temp):
                err_0 = new_err
            # otherwise reject the weight change and repeat for all 21 weights
            else:
                w[i] = w_og[i]
    # prints the iteration number every line
    print("iteration", ite+1, "\nwrong :(\n\n\n")
    # add the current error value to the array called error
    error.append(err_0)
# if the algorithm reached the correct output, this prints the final values
if correct == 8:
    print("done! :)\n\nweights are:", w)
# if the algorithm doesnt work for 5000 iterations, will print the final values anyway
else:
    print("incorrect :(\n\nweights are:", w)

# find the final error and output of the final weights
correct, err_0, out = ntest()
# prints a table of the expected result, the round of the output, and the actual output for all 8 rows of the truth table
print("\nexpected   rounded output         output")
for i in range(8):
    print("   ", t[i][3], "          ", round(out[i]), "        ", out[i])
print("iterations=", ite+1)
print("outputs correct=", correct, "/8")
print("final error=", err_0, "\n-------------------------------------")
# plots a graph of the error as a function of the iterations
x = np.linspace(0, len(error), len(error))
plt.plot(x, error, color='orchid')
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title(f"Change In Error From The Metropolis Algorithm. temp={temp}, stepsize={stepsize}, k={k}")
plt.show()
