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
        
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.gamma1 = nn.Parameter(torch.tensor(1.0))
        self.betta1 = nn.Parameter(torch.tensor(1.0))
        self.delt1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))
        self.gamma2 = nn.Parameter(torch.tensor(1.0))
        self.betta2 = nn.Parameter(torch.tensor(1.0))
        self.delt2 = nn.Parameter(torch.tensor(1.0))
        
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
        x1 = x1#F.normalize(x1)
        x2 = x2#F.normalize(x2)
        
        
        mean_x = self.mean_module(x2)
        g2 = 2048
        g = 512
        g3 = g +g2
        oo = 1.#000#0.5#000.#000.#00000.
        covar_x1 = self.covar_module1(x1[:,0:1*g], x2[:,0:1*g]).add_jitter(jitter_val=self.jitter_val).to_dense()#.evaluate() 
        covar_x5 = self.covar_module5(x1[:,1*g:1*g3], x2[:,1*g:1*g3]).add_jitter(jitter_val=self.jitter_val).to_dense()                
        covar_x2 = self.covar_module2(x1[:,1*g3:g3+g], x2[:,1*g3:g3+g]).add_jitter(jitter_val=self.jitter_val).to_dense()         
        covar_x6 = self.covar_module6(x1[:,g3+g:2*g3], x2[:,g3+g:2*g3]).add_jitter(jitter_val=self.jitter_val).to_dense()        
        covar_x3 = self.covar_module3(x1[:,2*g3:2*g3+g], x2[:,2*g3:2*g3+g]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x7 = self.covar_module7(x1[:,2*g3+g:3*g3], x2[:,2*g3+g:3*g3]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x4 = self.covar_module4(x1[:,3*g3:3*g3+g], x2[:,3*g3:3*g3+g]).add_jitter(jitter_val=self.jitter_val).to_dense()
        covar_x8 = self.covar_module8(x1[:,3*g3+g:4*g3], x2[:,3*g3+g:4*g3]).add_jitter(jitter_val=self.jitter_val).to_dense()
      
        '''
        unc1 = x1[:, 4*g:5*g].unsqueeze(0)
        #print(unc1)
        unc2 = x1[:, 5*g:6*g].unsqueeze(0)
        unc3 = x1[:, 6*g:7*g].unsqueeze(0)
        unc4 = x1[:, 7*g:8*g].unsqueeze(0)
        unc = unc1
        unc = torch.cat((unc, unc2), dim=0)
        unc = torch.cat((unc, unc3), dim=0)
        unc = torch.cat((unc, unc4), dim=0)
        unc = unc.mean(0)
        
        #print(unc.size())
        covar_x10 = self.covar_module10(unc, unc).add_jitter(jitter_val=self.jitter_val).evaluate()
        

        unc5 = x1[:, 4*g3+g:5*g3].unsqueeze(0)
        #print(unc1)
        unc6 = x1[:, 5*g3+g:6*g3].unsqueeze(0)
        unc7 = x1[:, 6*g3+g:7*g3].unsqueeze(0)
        unc8 = x1[:, 7*g3+g:8*g3].unsqueeze(0)
        unc = unc5
        unc = torch.cat((unc, unc6), dim=0)
        unc = torch.cat((unc, unc7), dim=0)
        unc = torch.cat((unc, unc8), dim=0)
        unc = unc.mean(0)
        
        #print(unc.size())
        #covar_x9 = self.covar_module9(unc, unc).add_jitter(jitter_val=self.jitter_val).evaluate()
        '''
        self.c1 = self.kerneltype(oo*(x1[:, 4*g3:4*g3+g]), ktype='', dim=0)+ 0.01
        self.c2 = self.kerneltype(oo*(x1[:, 5*g3:5*g3+g]), ktype='', dim=0)+ 0.01
        self.c3 = self.kerneltype(oo*(x1[:, 6*g3:6*g3+g]), ktype='', dim=0)+ 0.01
        self.c4 = self.kerneltype(oo*(x1[:, 7*g3:7*g3+g]), ktype='', dim=0)+ 0.01

        self.c5 = self.kerneltype(oo*(x1[:, 4*g:5*g]+ 0.01), ktype='', dim=0)
        self.c6 = self.kerneltype(oo*(x1[:, 5*g:6*g]+ 0.01), ktype='', dim=0)+ 0.01
        self.c7 = self.kerneltype(oo*(x1[:, 6*g3+g:7*g3]), ktype='', dim=0)+ 0.01
        self.c8 = self.kerneltype(oo*(x1[:, 7*g3+g:8*g3]), ktype='', dim=0)+ 0.01
                                        
        oo1 = 1.0
       
        if not self._initialized:
         with torch.no_grad():
         
          if self.kernel_function[0] == 'UaARDRBFKernel':
             self.covar_module1.base_kernel.lengthscale.copy_(self.covar_module1.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c1).mean(0))
          if self.kernel_function[1] == 'UaARDRBFKernel':
            
            self.covar_module2.base_kernel.lengthscale.copy_(self.covar_module2.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c2).mean(0))
          if self.kernel_function[2] == 'UaARDRBFKernel':
            
            self.covar_module3.base_kernel.lengthscale.copy_(self.covar_module3.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c3).mean(0))
          if self.kernel_function[3] == 'UaARDRBFKernel':
            
            self.covar_module4.base_kernel.lengthscale.copy_(self.covar_module4.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c4).mean(0))
          
 
          if self.kernel_function[4] == 'UaARDRBFKernel':
           
            self.covar_module5.base_kernel.lengthscale.copy_(self.covar_module5.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c5).mean(0))
          if self.kernel_function[5] == 'UaARDRBFKernel':
            
            self.covar_module6.base_kernel.lengthscale.copy_(self.covar_module6.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c6).mean(0))
          if self.kernel_function[6] == 'UaARDRBFKernel':
           
            self.covar_module7.base_kernel.lengthscale.copy_(self.covar_module7.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c7).mean(0))
          if self.kernel_function[7] == 'UaARDRBFKernel':
           
            self.covar_module8.base_kernel.lengthscale.copy_(self.covar_module8.base_kernel.raw_lengthscale_constraint.inverse_transform(oo1*self.c8).mean(0))
        self._initialized = True
        

        covar_x = self.b*(self.alpha1*covar_x1 + self.gamma1*covar_x2 + self.betta1*covar_x3 + self.delt1*covar_x4)+ self.a*(self.alpha2*covar_x5 + self.gamma2*covar_x6 + self.betta2*covar_x7 + self.delt2*covar_x8)#* covar_x9
        #covar_x = covar_x1 + covar_x2 + covar_x3 + covar_x4 #* covar_x9
        
        
        return mean_x, covar_x

    def _set_params(self, outputscale=8., lengthscale=1.):
        
        self.covar_module1.outputscale = 1*outputscale 
        self.covar_module2.outputscale = 1*outputscale 
        self.covar_module3.outputscale = 1*outputscale     
        self.covar_module4.outputscale = 1*outputscale  

        self.covar_module5.outputscale = 1*outputscale
        self.covar_module6.outputscale = 1*outputscale 
        self.covar_module7.outputscale = 1*outputscale 
        self.covar_module8.outputscale = 1*outputscale     
        #self.covar_module9.outputscale = 1*outputscale  
     
class OneClassGPModel(GPModel):
    def __init__(self, kernel_function, jitter_val=1e-3):
        super(OneClassGPModel, self).__init__(jitter_val)

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel_function = kernel_function
        g2 = 2048
        g = 512
       
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
            self.ker_fun4 = kernels.RBFKernel(ard_num_dims=g, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[4] == "ARDRBFKernel" or  kernel_function[4] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun5 = kernels.RBFKernel(ard_num_dims=g2, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[5] == "ARDRBFKernel"or  kernel_function[5] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun6 = kernels.RBFKernel(ard_num_dims=g2, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[6] == "ARDRBFKernel"or  kernel_function[6] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun7 = kernels.RBFKernel(ard_num_dims=g2, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[7] == "ARDRBFKernel"or  kernel_function[7] == "UaARDRBFKernel":
            # impose length scale of at least 1e-2
            self.ker_fun8 = kernels.RBFKernel(ard_num_dims=g2, lengthscale_constraint=constraints.GreaterThan(1e-2))
        if kernel_function[8] == "ARDRBFKernel"or  kernel_function[8] == "UaARDRBFKernel":
        #    # impose length scale of at least 1e-2
            self.ker_fun9 = kernels.RBFKernel(ard_num_dims=g2, lengthscale_constraint=constraints.GreaterThan(1e-2))
        
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
        self.covar_module6 = gpytorch.kernels.ScaleKernel(self.ker_fun6)
        self.covar_module7 = gpytorch.kernels.ScaleKernel(self.ker_fun7)
        self.covar_module8 = gpytorch.kernels.ScaleKernel(self.ker_fun8)
        self.covar_module9 = gpytorch.kernels.ScaleKernel(self.ker_fun9)
       
       
