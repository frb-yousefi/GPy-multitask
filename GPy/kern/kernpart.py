# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


class kernpart(object):
    def __init__(self,D):
        """
        The base class for a kernpart: a positive definite function which forms part of a kernel

        :param D: the number of input dimensions to the function
        :type D: int

        Do not instantiate.
        """
        self.D = D
        self.Nparam = 1
        self.name = 'unnamed'

    def get_param(self):
        raise NotImplementedError
    def set_param(self,x):
        raise NotImplementedError
    def get_param_names(self):
        raise NotImplementedError
    def K(self,X,X2,target):
        raise NotImplementedError
    def Kdiag(self,X,target):
        raise NotImplementedError
    def dK_dtheta(self,partial,X,X2,target):
        raise NotImplementedError
    def dKdiag_dtheta(self,partial,X,target):
        raise NotImplementedError
    def psi0(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dtheta(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,partial,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,partial,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def dK_dX(self,X,X2,target):
        raise NotImplementedError


