import sympy as sp
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow_probability as tfp

import MLGeometry as mlg
from MLGeometry import bihomoNN as bnn
import math

import network_functions as nf
import hermitian_met as hm

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(42)
tf.random.set_seed(42)

# Matplotlib options

plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['figure.figsize'] = [4, 2]
plt.rcParams['text.usetex'] = True





# Define equation for elliptic curve
z0, z1, z2 = sp.symbols('z0, z1, z2')
Z = [z0, z1, z2]
a, b =  -1, 1
f = z0**3 + a * (z0**2)*z2 - (z1**2)*z2 + b*(z2**3)


# Create training and testing set
n_pairs = 10000#100 #10000
HS_train = mlg.hypersurface.Hypersurface(Z, f, n_pairs)
HS_test = mlg.hypersurface.Hypersurface(Z, f, n_pairs)
train_set = mlg.tf_dataset.generate_dataset(HS_train)
test_set = mlg.tf_dataset.generate_dataset(HS_test)
train_set = train_set.shuffle(HS_train.n_points).batch(1000)
test_set = test_set.shuffle(HS_test.n_points).batch(1000)
kahler_pot = nf.Kahler_potential(3)
print("Network defined")




# Train and save Kahler potential if necessary:

# optimizer = tf.keras.optimizers.Adam()
# loss_func = nf.weighted_MAPE #mlg.loss.weighted_MAPE
# max_epochs = 50 #200 #500
# epoch = 0
# print("Beginning training")
# while epoch < max_epochs:
#     epoch = epoch + 1
#     for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
#         with tf.GradientTape() as tape:
#             det_omega = nf.volume_form(points, Omega_Omegabar, mass, restriction, kahler_pot)
#             # print("det_omega shape: ", tf.shape(det_omega))
#             # print("Omega_Omegabar shape: ", tf.shape(Omega_Omegabar))
#             loss = loss_func(Omega_Omegabar, det_omega, mass)
#             grads = tape.gradient(loss, kahler_pot.trainable_weights)
#         optimizer.apply_gradients(zip(grads, kahler_pot.trainable_weights))
#     if epoch % 50 == 0:
#         print("epoch %d: loss = %.5f" % (epoch, loss))

# kahler_pot.save("ec_kahler_pot")
# sigma_test = nf.cal_total_loss(test_set, mlg.loss.weighted_MAPE, kahler_pot)
# E_test = nf.cal_total_loss(test_set, mlg.loss.weighted_MSE, kahler_pot)
# print("sigma_test = %.5f" % sigma_test)
# print("E_test = %.7f" % E_test)


kahler_pot = tf.keras.models.load_model("ec_2_kahler_pot")


# Define and train model for hermitian metric
# Define and train model for hermitian metric
hermitian_met = hm.Hermitian_metric_O4(3, 1, [10,100])
print("Hermitian metric G network defined")
c1 = tf.constant(4., shape=(1000), dtype=tf.complex64)
optimizer = tf.keras.optimizers.Adam()
loss_func = nf.weighted_loss 
max_epochs = 5 
epoch = 0
while epoch < max_epochs:
    epoch = epoch + 1
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
        with tf.GradientTape() as tape:
            curv_tr = nf.curvature_trace(points, kahler_pot, hermitian_met, mass)
            #print("calculate curv_tr: ", curv_tr)
            #print("c1: ", c1)
            loss = loss_func(c1, curv_tr, mass) 
            # print("calculated loss: ", loss)
            grads = tape.gradient(loss, hermitian_met.trainable_weights)
        optimizer.apply_gradients(zip(grads, hermitian_met.trainable_weights))
    if epoch % 50 == 0:
        print("epoch %d: loss = %.7f" % (epoch, loss))
    hermitian_met.append_loss(loss)

total_test_loss = nf.cal_total_loss(test_set, hermitian_met, kahler_pot, nf.weighted_loss, c1)
print("Total testing loss: ", total_test_loss)

nf.plot_train_hist(hermitian_met, y_scale='linear')
