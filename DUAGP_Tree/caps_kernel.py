#raise Exception("TEST FILE")
from gpytorch import kernels
from gpytorch import constraints
import gpytorch
import torch
from torch import nn
import torch.nn.functional as F
import gpytorch

class GPModel(gpytorch.Module):
   def __init__(self, jitter_val=1e-3):
        super().__init__()
        # mean and cov functions
        self.jitter_val = jitter_val
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.betta = nn.Parameter(torch.tensor(1.0))
        self.delt = nn.Parameter(torch.tensor(1.0))

    #def entropy(self, x, dim=-1):
    #   log_p = torch.log_softmax(x, dim=dim)
    #   p = torch.exp(log_p)
    #   return -torch.sum(p * log_p, dim=dim)
   def entropy(self,f):
        eps = 1e-8
        enerGf = f**2
        Pf = torch.clamp(enerGf/(enerGf.sum() + eps), min=eps) 
        entropyf = (-Pf*torch.log(Pf + eps)) 
        return entropyf

   def kerneltype(self, f, ktype = 'UA1',dim=0):
       
       if ktype == 'UA1':
         
         return torch.abs(torch.std(self.entropy(f), dim=0))
       elif ktype == 'UA2':
         feat = torch.std(f, dim)
         return self.entropy(feat)
       elif ktype == 'EA':
           return self.entropy(self.entropy(f))
       else: 
          return torch.std(f, dim)

   def forward(self, x1, x2=None):
        if x2 is None:
            
            x2 = x1
        
        # L2 normalization
        x1 = F.normalize(x1)
        x2 = F.normalize(x2)
        
        
        mean_x = self.mean_module(x2)
        
        g = 3
        
        oo = 1.#1000.#000#0.5#000.#000.#00000.
        covar_x1 = self.covar_module1(x1[:,0:1*g], x2[:,0:1*g]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x2 = self.covar_module2(x1[:,1*g:2*g], x2[:,1*g:2*g]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x3 = self.covar_module3(x1[:,2*g:3*g], x2[:,2*g:3*g]).add_jitter(jitter_val=self.jitter_val).to_dense()        
        covar_x4 = self.covar_module4(x1[:,3*g:3*g+2], x2[:,3*g:3*g+2]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x5 = self.covar_module5(x1[:,3*g+2:5*g], x2[:,3*g+2:5*g]).add_jitter(jitter_val=self.jitter_val).to_dense()

        oo1 = 1.0
        
        c1 = self.kerneltype(oo*x1[:, 0:1*g], ktype='UA2', dim=0)+ 0.01
        c2 = self.kerneltype(oo*x1[:,1*g:2*g], ktype='UA2', dim=0)+ 0.01
        c3 = self.kerneltype(oo*(x1[:, 2*g:3*g]), ktype='UA1', dim=0)+ 0.01
        c4 = self.kerneltype(oo*(x1[:, 3*g:3*g+2]), ktype='EA',dim=0)+ 0.01
        c5 = self.kerneltype(oo*(x1[:, 3*g+2:5*g]), ktype='EA', dim=0)+ 0.01
        if not self._initialized:
          with torch.no_grad():
               if self.kernel_function[0] == 'UaARDRBFKernel':
                  self.covar_module1.base_kernel.lengthscale.copy_(self.covar_module1.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*c1).mean(0))
               if self.kernel_function[1] == 'UaARDRBFKernel':           
                  self.covar_module2.base_kernel.lengthscale.copy_(self.covar_module2.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*c2).mean(0))
               if self.kernel_function[2] == 'UaARDRBFKernel':            
                  self.covar_module3.base_kernel.lengthscale.copy_(self.covar_module3.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*c3).mean(0))
               if self.kernel_function[3] == 'UaARDRBFKernel':            
                  self.covar_module4.base_kernel.lengthscale.copy_(self.covar_module4.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*c4).mean(0))
               if self.kernel_function[4] == 'UaARDRBFKernel':
                  self.covar_module5.base_kernel.lengthscale.copy_(self.covar_module5.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*c5).mean(0))
        
        self._initialized = True
        covar_x = self.alpha * covar_x1 + self.gamma * covar_x2 + self.betta * covar_x3 + self.delt * covar_x4 * covar_x5
       
        return mean_x, covar_x

   def _set_params(self, outputscale=8., lengthscale=1.):
        
        self.covar_module1.outputscale = 1*outputscale 
        self.covar_module2.outputscale = 1*outputscale 
        self.covar_module3.outputscale = 1*outputscale     
        self.covar_module4.outputscale = 1*outputscale  
        self.covar_module5.outputscale = 1*outputscale
        

      

     
class OneClassGPModel(GPModel):
    def __init__(self, kernel_function, jitter_val=1e-3):
        super(OneClassGPModel, self).__init__(jitter_val)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_function = kernel_function
        
        g = 3 
        
        if kernel_function[0] == "ARDRBFKernel"or  kernel_function[0] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun1 = kernels.RBFKernel(ard_num_dims=g, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[1] == "ARDRBFKernel"or  kernel_function[1] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun2 = kernels.RBFKernel(ard_num_dims=g, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[2] == "ARDRBFKernel"or  kernel_function[2] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun3 = kernels.RBFKernel(ard_num_dims=g, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[3] == "ARDRBFKernel"or  kernel_function[3] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun4 = kernels.RBFKernel(ard_num_dims=g-1, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[4] == "ARDRBFKernel" or  kernel_function[4] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun5 = kernels.RBFKernel(ard_num_dims=g+1, lengthscale_constraint=constraints.GreaterThan(1e-2))
        
        
        if kernel_function[0] == "PeriodicKernel":
             self.ker_fun1 = kernels.PeriodicKernel()
        if kernel_function[1] == "PeriodicKernel":
             self.ker_fun2 = kernels.PeriodicKernel()
        if kernel_function[2] == "PeriodicKernel":
             self.ker_fun3 = kernels.PeriodicKernel()
        if kernel_function[0] == "MaternKernel":
            self.ker_fun1 = kernels.MaternKernel()
        if kernel_function[1] == "MaternKernel":
            self.ker_fun2 = kernels.MaternKernel()
        if kernel_function[2] == "MaternKernel":
            self.ker_fun3 = kernels.MaternKernel()
        if kernel_function[3] == "MaternKernel":
            self.ker_fun4 = kernels.MaternKernel()
        if kernel_function[4] == "MaternKernel":
            self.ker_fun5 = kernels.MaternKernel()
        if kernel_function[0] == "LinearKernel":
            self.ker_fun1 = kernels.LinearKernel()
        
        self._initialized = False
        self.covar_module1 = gpytorch.kernels.ScaleKernel(self.ker_fun1)
        self.covar_module2 = gpytorch.kernels.ScaleKernel(self.ker_fun2)
        self.covar_module3 = gpytorch.kernels.ScaleKernel(self.ker_fun3)
        self.covar_module4 = gpytorch.kernels.ScaleKernel(self.ker_fun4)
        self.covar_module5 = gpytorch.kernels.ScaleKernel(self.ker_fun5)
        
       
