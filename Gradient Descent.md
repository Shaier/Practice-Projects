```python
import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([6,8,10,12,15])

def gradient_descent(x,y):
    #initial conditions for m and b (mx+b is the line. The idea is to minimize the errors (sum of square errors): y_i-y_predicted (y_i-(mx+b))
    m_curr=b_curr=0
    #how many itirations
    iterations=1000
    #to calculate the partial derivative of x we need to divide by how many items are in x - so we set 'n' to be that value (we assume x and y are equal length here)
    #the partial derivative gives us the direction (slope) of the line
    n=len(x)
    #learning rate is a way for us to decrease the size of each step that we take until we reach a point that the steps are close to zero (that's when
    #the slope is close to zero and we found the min)
    learning_rate=0.083
    #start finding the best line
    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=(1/n) * sum([val**2 for val in (y-y_predicted)])
        m_derivative=-(2/n)*sum(x*(y-y_predicted))
        b_derivative=-(2/n)*sum(y-y_predicted)
        #once we get the slope for b and m we can adjust our m's, b's
        m_curr = m_curr - learning_rate * m_derivative
        b_curr = b_curr - learning_rate * b_derivative
        print("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

gradient_descent(x,y)
```
