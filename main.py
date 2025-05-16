import numpy as np
import matplotlib.pyplot as plt


def least_error(data):
    """ Prints the constants with the least error """
    best = data[0]
    for i in data:
        if i["error"] < best["error"]:
            best = i
    print(best)


def MS_error(m, x, y, w, b):
    """ Calculates the Mean Square error of the prediction """
    error = 0.0
    for i in range(m):
        error += ((x[i] * w) + b - y[i]) ** 2
    error = error / m
    return error


def derivative(m, x, y, w, b):
    """ Returns derivative of error w.r.t 'w' & 'b' """
    error_w = 0.0
    error_b = 0.0
    for i in range(m):
        error_w += ((x[i] * w) + b - y[i]) * 2 * x[i]
        error_b += ((x[i] * w) + b - y[i]) * 2
    error_w = error_w / m  # Mean square error w.r.t 'w'
    error_b = error_b / m  # Mean square error w.r.t 'b'

    return error_w, error_b


def new_const(m, x, y, w, b, alpha):
    """ Uses the derivatives of error to provide new constants """
    dw, db = derivative(m, x, y, w, b)
    new_w = w - (alpha * dw)  # provides the correction to 'w'
    new_b = b - (alpha * db)  # provides the correction to 'w'
    return new_w, new_b


x_in = np.array([1, 2, 3, 4])
y_out = np.array([12, 33, 44, 55])

w_arb = 0.0  # slope of the linear model
b_arb = 0.0  # bias of the linear model

m = x_in.shape[0]  # no. of rows of 'x' input matrix
learning_constant = 0.6e-1  # learning constant by which the model progress

num_iter = 100000  # no. of total testing iterations
num_breaks = 10  # no. of breaks
break_gap = num_iter / num_breaks  # gap between each break
breaks = []  # iter at which each break occurs
data = []  # data for each break

for i in range(num_breaks):
    breaks.append(i * break_gap)

# starting the learning process
for n in range(num_iter):
    w_arb, b_arb = new_const(m=m, x=x_in, y=y_out, w=w_arb, b=b_arb, alpha=learning_constant)

    if n in breaks:
        error = MS_error(m=m, x=x_in, y=y_out, w=w_arb, b=b_arb)
        data.append({"iter": n, "w": w_arb, "b": b_arb, "error": error})  # records the analysis of current model

least_error(data=data)  # printing the best model

# ploting the model
# x = np.linspace(x_in[0], x_in[m - 1], m * 10)
# y = w_arb * x + b_arb
#
# fig, ax = plt.subplots()
# ax.plot(x, y)
#
# plt.scatter(x_in, y_out, c="black")
#
# plt.show()
