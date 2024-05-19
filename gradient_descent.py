class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        '''
            task:
            Minimize the function f(x) -> x^2 using gradient descent
            Take the minimizer x = 0

            input: 
            - `iterations`: the number of iterations to perform gradient descent. `iterations >= 0`
            - `learning_rate`: the learning rate for gradient descent. `1 > learning_rate > 0`
            - `init`: the initial guess for the minimizer. `init != 0`
            
            output:
            Result to 5 decimal places using Python's round() function
            
            steps:
            1. Given a function f(x) -> x^2
            2. Then the derivative, d of f(x) -> 2x
            3. Compute the new derivative with every iteration
            4. Subtract learning_rate * derivative from the initial guess with every iteration
        '''
        for i in range(iterations):
            d = 2 * init
            init -= learning_rate * d
        
        return round(init, 5)