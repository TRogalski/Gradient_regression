'''Gradient descent for linear regression

repeat until convergence: {

theta_i = theta_i - alfa * 1/m * Ei_m(h(xi)-yi)

}

Notes:
-All parameters ought to be updated simultaneously
-Above form comes from partial derivatives over the cost function:

J(theta) = 1/2m * Ei_m(h(xi)-yi)^2

Which means we basically want to minimize square distances between
our model outputs and observed values.

More general form of the algoritm:

repeat until convergence: {

theta_i = theta_i - alfa partial_deriv(theta_i)J(theta) 

}

h(x) is just function representing our model.

The regression parameters for smaller problems can be derived analyticcally
from normal equation:

theta^=(X'X)^-1X'y

X - data matrix
y - output vector
theta^ - parameter vector

This may not be a good solution when we have too many observations as reversing the
X matrix becomes a very expensive action.


Gradient descent cons:
-have to set alfa (we may choose too high or too low one making it not reaching or overshoot optima)
-may get stuck in a local optima when we are using it with non-convex functions


Important!:
-apply feature scalling when features are not on a similar scale.
This will help alghoritm find the optima. You can do this by:

feature scalling: get every feature approximately -1<=xi<=1 range.
(we divide input values by the range (i. e. the maximum value minus the minimum value)
of the input variable, resulting in a new range of just 1.

mean normalization: it involves subtracting the average value of an input variable
from the values for that input variable resulting in a new average value
for the input variable of just zero.

both techniques can be implemented at once

-declare convergence if J(theta) decreases by less than 10^-3 in one iteration
'''

# coding: utf-8

# In[130]:


import numpy as np
import matplotlib.pyplot as plt
import copy
from numpy.linalg import inv


x=np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [10.3,10.3,10.1,9.3,8.4,7.3,8.4,7.9,7.6,7.6,6.9,7.4,8.1,7.0,6.5,5.8]])
y=np.array([183800,183200,174900,173500,172900,173200,173200,169700,174500,177900,
            188100,203200,230200,258200,309800,329800])

plt.plot(y)
plt.show()


# In[131]:


len(y)


# In[132]:


np.transpose(x)


# In[133]:


np.transpose(y)


# In[134]:


#normal equation parameters

normal_eq_theta=np.matmul(inv(np.matmul(x,np.transpose(x))),np.matmul(x,y))
normal_eq_theta


# In[135]:


np.matmul(np.transpose(x),normal_eq_theta)


# In[139]:


alfa=0.01
convergence=False
theta=np.array([0,0])
new_theta=([0,0])
buka=[]

cost_previous=1/(2*len(y))*np.sum(np.power((np.matmul(np.transpose(x),theta)-y),2))
while not convergence:
    for i in range(len(theta)):
        new_theta[i]=theta[i]-alfa*1/len(y)*np.sum((np.matmul(np.transpose(x),theta)-y)*x[i])
    cost_current=1/(2*len(y))*np.sum(np.power((np.matmul(np.transpose(x),new_theta)-y),2))
    if abs(cost_previous-cost_current)<10^2:
        convergence=True
    else:
        cost_previous=1/(2*len(y))*np.sum(np.power((np.matmul(np.transpose(x),new_theta)-y),2))
    theta=np.copy(new_theta)
    buka.append(cost_current)
theta


# In[140]:


plt.plot(buka)


# In[141]:


plt.show()


# In[142]:


np.matmul(np.transpose(x),theta)-y


# In[105]:





