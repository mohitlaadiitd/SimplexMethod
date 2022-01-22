import numpy as np
import sys

'''
   The objective is to find the optimum value of the function c'X, constrained by A'X = b, X >= 0
   where A is an m*n matrix.
   The initial basic feasible solution is taken by setting variables x-1 to x-(n-m) as zero and 
   basic variables as last m variables. 

   - TABLEAU METHOD -
   The code develops a Python Class *Simplex* that takes three neccessary arguments, (A, b, c) and
   has methods defined inside it. The class allows us to find the optimized value using the 
   Simplex.Optimizer() function.
'''

class Simplex:

    '''
        A -> matrix of coeficients
        b -> RHS of constraints
        c -> coeffecients of variables in objective function
        m -> number of constraints/slack vars
        n -> number of vars
        vars -> array containing names of variables 1..n
        basic_vars -> array containing names of current basic vars in order
        non_basic_vars -> array containing names of current non-basic vars in order
        N -> matrix of non-basic vars' coefficients
        B -> matrix of basic vars' coefficients 
        xB -> current value of current basic vars
        xN -> current value of current non-basic vars
        cB -> coeffecients of basic vars in objective function
        cN -> coeffecients of non-basic vars in objective function
        multiple_sols -> boolean variable to see if problem has multiple optimal pts
    '''

    def __init__(self, A, b, c):
        
        self.A = A
        self.b = b
        self.c = c
        self.declare_initial_values() # calls this method to set initial values of problem
      
    # initializes attributes defined on self object
    def declare_initial_values(self):
        
        self.m, self.n = self.A.shape
        m, n = self.m, self.n
        self.vars = [i for i in range(1, n+1)]
        self.cN, self.cB = c[:n-m].T, c[n-m:].T
        self.non_basic_vars, self.basic_vars = self.vars[:n-m], self.vars[n-m:]
        self.N, self.B = np.array(A[:, :n-m]), np.array(A[:, n-m:])
        self.xN = np.zeros((n-m, 1))
        self.xB = np.linalg.inv(self.B).dot(self.b)
        self.multiple_sols = False
    
    # if all the values of cN_hat vector are positive(no further decrease in Z), optimality is reached
    def is_optimality_reached(self):
        
        return np.all(self.cN_hat >= 0)
    

    # if at optimality, there is a variable that has zero contribution while not being in basis, it 
    # can lead to an alternate optimal solution
    def check_multiple(self):
        
        for contribution in self.cN_hat:
            if contribution == 0:
                self.multiple_sols = True
                break 
        return

    # finds the variable not in basis that can decrease Z the most
    def find_entering_variable(self):
        
        return np.argmin(self.cN_hat)

    # A_t_hat contains the constraint coefficients corresponding to the entering variable
    # Ratio test will find the leaving variable, and determine the pivot entry
    # If all the A_t_hat's are <= 0, the problem is unbounded
    def find_leaving_variable(self, entering_variable):        
        
        A_t_hat = np.linalg.inv(self.B).dot(np.array(self.A[:, self.non_basic_vars[entering_variable]-1]).T)
        b_hat = np.linalg.inv(self.B).dot(self.b)
        
        min_ratio, min_ratio_index = float('inf'), -1
        for i in range(self.m):
            if A_t_hat[i] <= 0: continue
            ratio = b_hat[i]/A_t_hat[i]
            if ratio < min_ratio:
                min_ratio = ratio
                min_ratio_index = i 
        if min_ratio_index == -1:
            print("Unbounded Solution")
            return 

        leaving_variable = min_ratio_index 
        # print(f"{self.non_basic_vars[entering_variable]} enters and {self.basic_vars[leaving_variable]} leaves")
        return self.update_values(entering_variable, leaving_variable)
    
    # the values corresponding to the two concerned entering var and leaving var are interchanged
    def update_values(self, entering_variable, leaving_variable):
        
        leaving_variable_name = self.basic_vars[leaving_variable]
        self.basic_vars[leaving_variable] = self.non_basic_vars[entering_variable]
        self.non_basic_vars.pop(entering_variable)
        
        # the matrices corresponding to the non-basic vars need to be sorted according to the vars in self.non_basic_vars
        indx = 0
        for i in range(self.n-self.m-2):
            if self.non_basic_vars[i] < leaving_variable_name < self.non_basic_vars[i+1]:
                indx = i+1
        if indx == 0:
            if self.non_basic_vars[self.n-self.m-2] < leaving_variable_name:
                indx = self.n-self.m-1
        
        self.non_basic_vars.insert(indx, leaving_variable_name)
        
        self.cB[leaving_variable], self.cN[entering_variable] = self.cN[entering_variable], self.cB[leaving_variable]
        self.B[:, leaving_variable], self.N[:, entering_variable] = np.array(self.N[:, entering_variable]), np.array(self.B[:, leaving_variable])
        
        self.xB = np.linalg.inv(self.B).dot(self.b)
        
        self.cN[indx], self.cN[entering_variable] = self.cN[entering_variable], self.cN[indx]
        self.N[:, indx], self.N[:, entering_variable] = np.array(self.N[:, entering_variable]), np.array(self.N[:, indx])
   
    # this method iteratively finds out the optimum value
    def optimize(self):
        
        y = (self.cB.T.dot(np.linalg.inv(self.B))).T
        self.cN_hat = self.cN - self.N.T.dot(y) # contribution vector of non-basic vars to Z
        
        while not self.is_optimality_reached():
            
            entering_variable = self.find_entering_variable()
            self.find_leaving_variable(entering_variable)
            
            y = (self.cB.T.dot(np.linalg.inv(self.B))).T
            self.cN_hat = self.cN - self.N.T.dot(y)
        
        if self.is_optimality_reached(): self.check_multiple()
        self.output_answers()
    
    # logs the results and other info about the optimizer's solution
    def output_answers(self):
        
        self.xN = np.linalg.pinv(self.N).dot((self.b - self.B.dot(self.xB)))

        Z = self.cB.dot(self.xB) + self.cN.dot(self.xN)
        print(f"The optimal objective function value is {Z}")
        
        final_vals = [None for _ in range(self.n)]
        for var in range(self.m):
            final_vals[self.basic_vars[var]-1] = self.xB[var]

        for var in range(self.n-self.m):
            final_vals[self.non_basic_vars[var]-1] = self.xN[var]
        
        print(f"The optimal solution is {final_vals}")
        if self.multiple_sols: print("This LP has multiple optimal solutions")
        else: print("This LP has a unique optimal solution")


'''
    reading A, b, c matrices from txt_file passed through command line

    line 1 -> m, n
    line 2,...,m+1 -> rows of matrix A
    line m+2 -> RHS 'b' of constraint equation
    line m+3 -> coeffecients 'c' of vars in objective function
'''

txt_file = sys.argv[1]
lines, A = [], []
for line in open(txt_file):
    lines.append(line)

m, n = list(map(int, lines[0].split(' ')))
for i in range(1, m+1):
    A.append(list(map(int, lines[i].split(' '))))
b = list(map(int, lines[m+1].split(' ')))
c = list(map(int, lines[m+2].split(' ')))

A = np.array(A)
b = np.array(b).T
c = np.array(c).T

Optimizer = Simplex(A, b, c)
Optimizer.optimize() 