"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.params = dict()
        self.grads = dict()

        self.params['weight'] = np.random.normal(0, 0.0001, (out_features, in_features))
        self.params['bias'] = np.zeros((1, out_features))

        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros((1, out_features))
        
        ########################
        # END OF YOUR CODE    #
        #######################
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.input = x

        w = self.params['weight']
        b = self.params['bias']

        out = x @ w.T + b
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        w = self.params['weight']
        x = self.input

        self.grads['weight'] = dout.T @ x
        self.grads['bias'] = dout.T.sum(1)
        dx = dout @ w
        
        ########################
        # END OF YOUR CODE    #
        #######################
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        b = x.max(1, keepdims=True)
        out = np.exp(x-b) / np.exp(x-b).sum(1, keepdims=True)

        self.output = out
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        y = self.output

        dx = dout * y - np.einsum('in, in, ij -> ij', dout, y, y)
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        s, n = x.shape

        out = - (y * np.log(x)).sum() / s
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        s, n = x.shape

        dx = - (y / x) / s
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.input = x

        out = np.where(x >= 0, x, np.exp(x) - 1)

        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        x = self.input

        dx = dout * np.where(x >= 0, 1, np.exp(x))

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
