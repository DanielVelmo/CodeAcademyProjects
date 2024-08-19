## To run the porgram enter in Terminal the: streamlit run /filepath
import pandas as pd
import numpy as np
import streamlit as st
from math import sin, cos, log, pi, tan
import matplotlib.pyplot as plt

st.title('Derivatives by definition')
st.markdown('By Daniel Vélez Moyado')
st.markdown('It is known that the definition of a derivative from a function f(x) is defined as:')


latext = r'''
$$ 
f'(x)= \lim_{h\to0} \frac{f(x+h) - f(x)}{h}
$$
'''
st.write(latext)

st.markdown("In order to understand why this definition works, select a function from the selection box and change the h to see how as h approaches to 0 the function gets closer to the real derivative.")
col1, col2 = st.columns(2)
dx = col2.slider('h', 0.001, 5.0, step = 0.001)
function_selection = col1.selectbox('Select a function', ['sin(x)', 'x^2 * log(x)', 'tan(x)'])


def limit_derivative(f, x, h):
    """
    f: function to be differentiated 
    x: the point at which to differentiate f 
    h: distance between the points to be evaluated
    """
    y_value = []
    for x_val in x: 
        y_value.append((f(x_val+h) - f(x_val))/h)
    return y_value

# f1(x) = sin(x)
def f1(x):
    return sin(x)
# f2(x) = x^4
def f2(x):
    return pow(x, 4)
# f3(x) = x^2*log(x)
def f3(x):
    return tan(x)
# f1'(x) = cos(x)
def f1_p(x):
    return cos(x)
# f2'(x) = 4x^3
def f2_p(x):
    return 4 * pow(x, 3)
# f3'(x) = 2x*log(x) + 1/x*x^2
def f3_p(x):
    return  1/(cos(x)**2)


def plot_function(f) : 
    x_values =  np.linspace(-10, 10, 2000)
    y_values = []
    for x in x_values : 
        y_values.append(f(x))
    return x_values, y_values


x_val, y_val = (0,0)
fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(10)
ax.set_ylim([-10,10])
ax.set_xlim([-10,10])
ax.set_title('f\'(x) plot')
if function_selection == 'sin(x)' :
    x_val, y_val = plot_function(f1_p) 
    y_prime = limit_derivative(f1, x_val, dx)
elif function_selection == 'x^2 * log(x)' : 
    x_val, y_val = plot_function(f2_p) 
    y_prime = limit_derivative(f2, x_val, dx)
elif function_selection == 'x^2 * log(x)' : 
    x_val, y_val = plot_function(f2_p) 
    y_prime = limit_derivative(f2, x_val, dx)
elif function_selection == 'tan(x)' : 
    x_val, y_val = plot_function(f3_p) 
    y_prime = limit_derivative(f3, x_val, dx)
    


ax.plot(x_val, y_val, label = 'f\'(x) real')
ax.plot(x_val, y_prime, label = f'f\'(x) where h = {dx}')
ax.legend(loc = 1)
ax.set_ylabel('y')
ax.set_xlabel('x')
st.pyplot(fig)
