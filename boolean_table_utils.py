"""Helper Functions For Blast Module Boolean Functions Jupyter Notebook
C Matthew Digman 2024"""

import inspect
import string
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

def test_and(op_in):
    return test_match_2arg(lambda x, y: x and y, op_in, true_label = 'A And B', test_label='Your Function', correct_label='Correct?')

def test_or(op_in):
    return test_match_2arg(lambda x, y: x or y, op_in, true_label = 'A Or B', test_label='Your Function', correct_label='Correct?')

def test_nor(op_in):
    return test_match_2arg(lambda x, y: not (x or y), op_in, true_label = 'A Nor B', test_label='Your Function', correct_label='Correct?')

def test_xor(op_in):
    return test_match_2arg(lambda x, y: x != y, op_in, true_label = 'A Xor B', test_label='Your Function', correct_label='Correct?')

def test_nand(op_in):
    return test_match_2arg(lambda x, y: not (x and y), op_in, true_label = 'A Nand B', test_label='Your Function', correct_label='Correct?')

def test_iff(op_in):
    return test_match_2arg(lambda x, y: (x == y), op_in, true_label = 'A IFF B', test_label='Your Function', correct_label='Correct?')

def test_impliesRight(op_in):
    return test_match_2arg(lambda x, y: (not x or y), op_in, true_label = 'A -> B', test_label='Your Function', correct_label='Correct?')

def test_impliesLeft(op_in):
    return test_match_2arg(lambda x, y: (x or not y), op_in, true_label = 'A <- B', test_label='Your Function', correct_label='Correct?')

def test_AAndNotB(op_in):
    return test_match_2arg(lambda x, y: not (not x or y), op_in, true_label = 'A And (Not B)', test_label='Your Function', correct_label='Correct?')

def test_NotAAndB(op_in):
    return test_match_2arg(lambda x, y: not (x or not y), op_in, true_label = '(Not A) and B', test_label='Your Function', correct_label='Correct?')

def test_true(op_in):
    return test_match_2arg(lambda x, y: True, op_in, true_label = 'True', test_label='Your Function', correct_label='Correct?')

def test_false(op_in):
    return test_match_2arg(lambda x, y: False, op_in, true_label = 'False', test_label='Your Function', correct_label='Correct?')

def test_not(op_in):
    return test_match_2arg(lambda x: not x, op_in, true_label = 'A', test_label='Your Function', correct_label='Correct?')

def test_A(op_in):
    return test_match_2arg(lambda x, y: x, op_in, true_label = 'A', test_label='Your Function', correct_label='Correct?')

def test_B(op_in):
    return test_match_2arg(lambda x, y: y, op_in, true_label = 'B', test_label='Your Function', correct_label='Correct?')

def test_notA(op_in):
    return test_match_2arg(lambda x, y: not x, op_in, true_label = 'not A', test_label='Your Function', correct_label='Correct?')

def test_notB(op_in):
    return test_match_2arg(lambda x, y: not y, op_in, true_label = 'not B', test_label='Your Function', correct_label='Correct?')

def half_adder(x, y):
    return x and y, x != y

def full_adder(z, x, y):
    return (x and y) or ((x!=y) and z), (x != y)!=z

all_labels = [\
        'A',\
        'B',\
        'Not A',\
        'Not B',\
        'A And B',\
        'A Or B',\
        'A Xor B',\
        'A Nor B',\
        'A Nand B',\
        'A -> B',\
        'A <- B',\
        'A IFF B',\
        'A And (Not B)',\
        '(Not B) And A',\
        'True',\
        'False',\
        ]

all_ops = [\
        lambda x, y: x,\
        lambda x, y: y,\
        lambda x, y: not x,\
        lambda x, y: not y,\
        lambda x, y: x and y,\
        lambda x, y: x or y,\
        lambda x, y: x != y,\
        lambda x, y: not (x or y),\
        lambda x, y: not (x and y),\
        lambda x, y: not x or y,\
        lambda x, y: x or not y,\
        lambda x, y: x == y,\
        lambda x, y: x and not y,\
        lambda x, y: y and not x,\
        lambda x, y: True,\
        lambda x, y: False,\
    ]

def gen_2_argument():
    A_text = np.array([True, True, False, False])
    B_text = np.array([True, False, True, False])

    loc_text = []

    for itrc in range(len(all_ops)):
        op_text = np.zeros(A_text.size,dtype=np.bool_)
        for itrx in range(A_text.size):
            op_text[itrx] = all_ops[itrc](A_text[itrx], B_text[itrx])

        loc_text.append(op_text)
    return loc_text

# generate all of the two argument maps                   
all_text = gen_2_argument()

def test_match_2arg(op_true, op_test, true_label = 'True', test_label='Test', correct_label='Correct?',input_label=None):
    if not callable(op_true) or not callable(op_test):
        raise ValueError('Inputs must be functions')

    n_par = len(inspect.getfullargspec(op_true).args)
    if n_par != len(inspect.getfullargspec(op_test).args):
        raise ValueError('Input functions must have same number of parameters')
    
    inputs_text = np.array(list(product([True,False],repeat=n_par)))
    n_inputs = inputs_text.shape[0]

    # check the number of return elements 
    res0 = op_true(*inputs_text[0])
    res1 = op_test(*inputs_text[0])
    if isinstance(res0,np.bool_) or isinstance(res0,bool):
        n_res0 = 1
        if not (isinstance(res1,np.bool_) or isinstance(res1,bool)):
            raise ValueError('Input Functions must have same number of return elements')
    else:
        if not hasattr(res0,'__len__') or not hasattr(res1,'__len__')  or len(res1) != len(res0):
            raise ValueError('Input Functions must have same number of return elements')    
        else:
            n_res0 = len(res0)

    true_text = np.zeros((n_inputs,n_res0),dtype=np.bool_)
    test_text = np.zeros((n_inputs,n_res0),dtype=np.bool_)

    for itrx in range(n_inputs):
        true_text[itrx] = op_true(*inputs_text[itrx])
        test_text[itrx] = op_test(*inputs_text[itrx])

    correct_text = true_text == test_text

    cell_labels = []
    if input_label is None:
        for itr in range(n_par):
            cell_labels.append(string.ascii_uppercase[itr])
    elif isinstance(input_label,str):
        for itr in range(n_par):
            cell_labels.append(input_label+' '+str(itr))
    else:
        for itr in range(n_par): 
            if len(input_label) <= itr:
                cell_labels.append('')
            else:
                cell_labels.append(input_label[itr])


    if n_res0 == 1:
        cell_labels.append(true_label)
        cell_labels.append(test_label)
        cell_labels.append(correct_label)
    else:
        for itr in range(n_res0):
            if isinstance(true_label,str):
                cell_labels.append(true_label+' '+str(itr))
            elif len(true_label) <= itr:
                cell_labels.append('')
            else:
                cell_labels.append(true_label[itr])
        for itr in range(n_res0):
            if isinstance(test_label,str):
                cell_labels.append(test_label+' '+str(itr))
            elif len(true_label) <= itr:
                cell_labels.append('')
            else:
                cell_labels.append(test_label[itr])
        for itr in range(n_res0):
            if isinstance(correct_label,str):
                cell_labels.append(correct_label+' '+str(itr))
            elif len(true_label) <= itr:
                cell_labels.append('')
            else:
                cell_labels.append(correct_label[itr])

    figsize = (6*len(cell_labels)/5,1.5*(n_inputs+1)/5)

    cell_text = np.vstack([inputs_text.T, true_text.T, test_text.T, correct_text.T])
    fig = gen_figure(cell_text, cell_labels, figsize=figsize)

    correct = np.all(correct_text)
    if correct:
        fig.set_facecolor('honeydew')
        fig.suptitle('Thats Right!',verticalalignment='top')
    else:
        if n_res0 == 1 and n_par == 2:
            res = np.all(test_text[:,0] == np.array(all_text), axis=1)
            if np.any(res):
                op_idx = np.argmax(res)
                label_loc = all_labels[op_idx]
            else:
                label_loc = "Huh???"

            title_label = 'Try Again! The operation you gave is called: ' + label_loc
        else:
            title_label = 'Try Again!'

        fig.set_facecolor('mistyrose')
        fig.suptitle(title_label, verticalalignment='top')

    return correct


def plot_2arg_table(op_in,op_label='Test',figsize=None):
    A_text = np.array([True, True, False, False])
    B_text = np.array([True, False, True, False])
    op_text = np.zeros(A_text.size,dtype=np.bool_)
    for itrx in range(A_text.size):
        op_text[itrx] = op_in(A_text[itrx], B_text[itrx])

    cell_labels = ['A', 'B', op_label]

    gen_figure([A_text, B_text, op_text], cell_labels, figsize=figsize)

def plot_1arg_table(op_in,op_label='Test',figsize=None):
    A_text = np.array([True, False])
    op_text = np.zeros(A_text.size,dtype=np.bool_)
    for itrx in range(A_text.size):
        op_text[itrx] = op_in(A_text[itrx])

    cell_labels = ['A', op_label]

    gen_figure([A_text, op_text], cell_labels, figsize=figsize)





def all_2_argument():
    figsize = (19.2,1.5)
    #figsize = None

    fig = gen_figure(all_text, all_labels, figsize=figsize)


def gen_figure(cell_text, cell_labels, figsize=None):
    cell_text = np.array(cell_text).T
    cell_color = np.zeros(cell_text.shape, dtype='object')
    cell_color[:] = 'lightcoral'
    cell_color[cell_text] = 'lightgreen'

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    ax.table(cellText=cell_text, cellColours=cell_color, colLabels=cell_labels, loc='center', cellLoc='center', rowLoc='center')

    fig.tight_layout()

    return fig

def ripple_adder(num1bool, num2bool):
    if not isinstance(num1bool,np.ndarray) or not isinstance(num2bool,np.ndarray):
        raise ValueError('Inputs must be array of booleans')

    if num1bool.size != num2bool.size:
        raise ValueError('Inputs must be the same size') 


    n_b = num1bool.size 

    res = np.zeros(num1bool.size,dtype=np.bool_)

    carry = False
    for itrb in range(n_b-1,-1,-1):
        carry, res[itrb] = full_adder(carry, num1bool[itrb], num2bool[itrb]) 

    return res


def unsigned2boolarray(number,n_bits=64):
    if not isinstance(n_bits,int) or n_bits <= 0:
        raise ValueError('Number of bits must be a positive integer')

    if not isinstance(number,int) or number < 0:
        raise ValueError('This function only takes non-negative integers')

    return (np.array([*format(number,'0'+str(n_bits)+'b')])=='1')[-n_bits:]

def boolarray2unsigned(bools):
    return int(''.join([format(bools[itr],'b') for itr in range(0,len(bools))]),2)

def ripple_adder_combos(adder_in, n_bits):
    if not isinstance(n_bits,int) or n_bits <= 0:
        raise ValueError('Number of bits must be a positive integer')
    
    reps = np.zeros((2**n_bits,n_bits),dtype=np.bool_)

    for itrx in range(0,2**n_bits):
        reps[itrx] = unsigned2boolarray(itrx,n_bits)

    n_tests = 2**n_bits*2**n_bits
    As = np.zeros(n_tests,dtype=np.int64)
    Bs = np.zeros(n_tests,dtype=np.int64)
    expecteds = np.zeros(n_tests,dtype=np.int64)
    results = np.zeros(n_tests,dtype=np.int64)
    
    itrt = 0 
    for itrx in range(0,2**n_bits):
        bool1 = reps[itrx]
        for itry in range(0,2**n_bits):
            bool2 = reps[itry]
            As[itrt] = itrx
            Bs[itrt] = itry
            expecteds[itrt] = (itrx+itry)%(2**n_bits)
            results[itrt] = boolarray2unsigned(adder_in(bool1,bool2))
            itrt = itrt+1
    return As,Bs,expecteds,results

def test_ripple_adder(adder_in,bits_max=4):
    if not isinstance(bits_max,int) or bits_max <= 0:
        raise ValueError('Number of bits must be a positive integer')

    pass_bits = np.zeros(bits_max,dtype=np.bool_)

    for itrb in range(1,bits_max+1):
        As, Bs, expecteds, results = ripple_adder_combos(adder_in, itrb)
        pass_bits[itrb-1] = np.all(expecteds == results)

    return np.all(pass_bits), pass_bits
    

def gen_adder_figure(adder_in, n_bits=3):
    As, Bs, expecteds, results = ripple_adder_combos(adder_in,n_bits)
    correct = expecteds == results

    figsize = (8,3.5*(n_bits**2+1)/5) 

    cell_text = np.array([As,Bs,expecteds,results]).T
    cell_color = np.zeros(cell_text.shape, dtype='object')
    cell_color[:] = 'white'
    cell_color[:,3] = 'lightcoral'
    cell_color[correct,3] = 'lightgreen'

    cell_labels = ['A (base 10)','B (base 10)','Expected Sum (base 10)','Test Sum (base 10)']

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.axis('tight')
    ax.axis('off')

    ax.table(cellText=cell_text, cellColours=cell_color, colLabels=cell_labels, loc='center', cellLoc='center', rowLoc='center')


    all_correct = np.all(correct)

    if all_correct:
        fig.set_facecolor('honeydew')
        title_label = 'Thats Right!'
    else:
        fig.set_facecolor('mistyrose')
        title_label = 'Try Again!'

    fig.suptitle(title_label, verticalalignment='top')

    #fig.tight_layout()
    fig.subplots_adjust()

    return fig

def plot_and_table():
    return plot_2arg_table(lambda x, y: x and y,op_label='And',figsize=(3.2,1.5))

def plot_andor_table():
    fig1 = plot_2arg_table(lambda x, y: x and y,op_label='And',figsize=(3.2,1.5))
    fig2 = plot_2arg_table(lambda x, y: x or y,op_label='Or',figsize=(3.2,1.5))

def plot_not_table():
    return plot_1arg_table(lambda x: not x,op_label='Not A',figsize=(6*3/5+0.2,1.5*3/5+0.5))

def test_half_adder(half_adder_in):
    return test_match_2arg(half_adder, half_adder_in, [r"$C_{out}$ Intended",'S Intended'], [r"$C_{out}$ Test",'S Test'],[r"$C_{out}$ Correct?",'S Correct?'],['A','B'])

def test_full_adder(full_adder_in):
    test_match_2arg(full_adder, full_adder_in, [r"$C_{out}$ Intended",'S Intended'], [r"$C_{out}$ Test",'S Test'],[r"$C_{out}$ Correct?",'S Correct?'],['A','B',r"$C_{in}$"])


