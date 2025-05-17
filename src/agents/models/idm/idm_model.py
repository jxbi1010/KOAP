import numpy as np
import torch
import torch.nn as nn
from vector_quantize_pytorch import FSQ
from agents.models.idm.sequence import MLPNetwork,LSTMBaseline,LSTMNetwork
# from agents.models.idm.byol import BYOL, LSTMEncoder

class BaselineModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, action_dim, mask_dim, backbone='mlp'):
        super(BaselineModel,self).__init__()
        self.backbone = backbone

        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) >2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'init input dim:{input_dim}')


        if self.backbone == 'mlp':
            self.network = MLPNetwork(input_dim, hidden_dims, action_dim, factorize=True)
        elif self.backbone == 'lstm':
            self.network =LSTMBaseline(input_dim, hidden_dims, action_dim)

        print(f'init baseline model with backbone:{self.backbone}')
    def forward(self, x):

        return self.network(x)


class VariationalAutoEncoder(nn.Module):
    # vae baseline method
    def __init__(self, input_dim,action_dim,hidden_dims,mask_dim, latent_dim,backbone='mlp'):
        super(VariationalAutoEncoder,self).__init__()
        self.backbone = backbone
        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) >2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'init input dim:{input_dim}')

        if backbone=='mlp':
            self.encoder = MLPNetwork(input_dim, hidden_dims, latent_dim, factorize=False)
            self.decoder = MLPNetwork(latent_dim, hidden_dims, input_dim,factorize=False)
            self.action_decoder = MLPNetwork(latent_dim, hidden_dims, action_dim, factorize=False)

        elif backbone=='lstm':
            self.encoder = LSTMNetwork(input_dim, hidden_dims, latent_dim)
            self.decoder = LSTMNetwork(latent_dim, hidden_dims, input_dim)
            self.action_decoder = LSTMNetwork(latent_dim, hidden_dims, action_dim)

        self.encoder_mu = MLPNetwork(latent_dim, [256, 256], latent_dim, factorize=True)
        self.encoder_var = MLPNetwork(latent_dim, [256, 256], latent_dim, factorize=True)

        print(f'init VariationalAutoEncoder with self.backbone:{self.backbone}, latent_dim:{latent_dim}, hidden_dims:{hidden_dims}')

    def reparametrization(self,mu,var):
        epsilon = torch.randn_like(var).to(mu.device)  # sampling epsilon
        z = mu + var * epsilon  # reparameterization trick
        return z

    def obs_encode(self,x_ref):
        # embed x to zx
        zx = self.encoder(x_ref)
        # use consecutive zx to compute za
        z_mu = self.encoder_mu(zx)
        z_var = self.encoder_var(zx)

        return zx,z_mu,z_var

    def forward(self,x_ref):

        zx,z_mu,z_var = self.obs_encode(x_ref)
        x_recon = self.decoder(zx)

        za = self.reparametrization(z_mu,torch.exp(0.5 * z_var))
        pred_act = self.action_decoder(za)

        return x_recon,pred_act,z_mu,z_var

    def inference(self,x_ref):

        _, z_mu, z_var = self.obs_encode(x_ref)

        za = self.reparametrization(z_mu, torch.exp(0.5 * z_var))
        pred_act = self.action_decoder(za)

        return pred_act

class AutoEncoderFSQ(nn.Module):
    #LAPO
    def __init__(self, input_dim, action_dim, hidden_dims, mask_dim, levels=[8, 5, 5, 5], latent_dim=4, backbone='mlp'):
        super(AutoEncoderFSQ, self).__init__()
        assert latent_dim % len(levels) == 0

        self.backbone = backbone
        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) >2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'[FSQ] init input dim:{input_dim}')


        if self.backbone == 'mlp':
            self.encoder = MLPNetwork(input_dim, hidden_dims, latent_dim,factorize=True)
            self.obs_decoder = MLPNetwork(latent_dim + input_dim, hidden_dims, input_dim, factorize=False)
            self.action_decoder = MLPNetwork(latent_dim+input_dim, [256,256], action_dim, factorize=False)

        elif self.backbone == 'lstm':
            self.encoder = LSTMBaseline(input_dim, hidden_dims, latent_dim)
            self.obs_decoder = LSTMNetwork(latent_dim + input_dim, hidden_dims, input_dim)
            self.action_decoder = LSTMNetwork(latent_dim+input_dim, [256],action_dim)

        self.vq = FSQ(levels=levels,dim=latent_dim,num_codebooks=2)

        print(f'init AutoEncoderFSQ model: latent_dim:{latent_dim}, levels:{levels}, backbone:{backbone}, hidden_dims:{hidden_dims}')

    def forward(self, x_ref):
        # x0 is history, x1 is current
        # e.g given seq_L = 16, predict 15 actions
        # use x1-x14 with actions a1-a14 to predict x2-x15 (x15 is the last in the seq)

        x_current = x_ref[:,1:-1]
        z = self.encoder(x_ref)
        vq_za, _ = self.vq(z)

        pred_x = self.obs_decoder(torch.cat((vq_za,x_current),dim=-1)) # use x till t to predict future x
        act = self.action_decoder(torch.cat((vq_za,x_current),dim=-1)) # use vq_za to predict a_t

        return pred_x, act

    def inference(self,x_ref):

        x_current = x_ref[:, 1:-1]
        z = self.encoder(x_ref)
        vq_za, _ = self.vq(z)

        act = self.action_decoder(torch.cat((vq_za,x_current),dim=-1)) # use x_t and z_t to predict a_t

        return act

class LatentDynamicsFSQ(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dims, mask_dim, levels=[8, 5, 5, 5], latent_dim=4, backbone='mlp'):
        super(LatentDynamicsFSQ, self).__init__()
        assert latent_dim % len(levels) == 0

        self.backbone = backbone

        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) >2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'[DynaFSQ] init input dim:{input_dim}')

        vq_dim = len(levels)*4

        if self.backbone == 'mlp':
            self.x_encoder = MLPNetwork(input_dim, hidden_dims, latent_dim,factorize=False)
            self.x_decoder = MLPNetwork(latent_dim, hidden_dims, input_dim,factorize=False)
            self.latent_action = MLPNetwork(latent_dim,[256,256,256],vq_dim,factorize=True)

        elif self.backbone == 'lstm':
            self.x_encoder = LSTMNetwork(input_dim, hidden_dims, latent_dim)
            self.x_decoder = LSTMNetwork(latent_dim, hidden_dims, input_dim)
            self.latent_action = LSTMBaseline(latent_dim,[256],vq_dim)

        self.a_decoder = MLPNetwork(vq_dim + input_dim, [256, 256], action_dim, factorize=False)
        self.latent_dynamics = MLPNetwork(vq_dim + latent_dim, hidden_dims, latent_dim, factorize=False)
        self.vq = FSQ(levels=levels,dim=vq_dim,num_codebooks=2)

        print(f'init LatentDynamicsFSQ model: latent_dim:{latent_dim}, levels:{levels}, backbone:{backbone}')

    def forward(self,x_ref):

        zx = self.x_encoder(x_ref)
        x_recon = self.x_decoder(zx)

        za = self.latent_action(zx)
        zx_current = zx[:, 1:-1, :]
        # zx_prime = zx[:,2:,:]

        vq_za,_ = self.vq(za)

        pred_zx_prime = self.latent_dynamics(torch.cat((vq_za,zx_current),dim=-1))

        x_current= x_ref[:,1:-1,:]
        a_pred = self.a_decoder(torch.cat((vq_za,x_current),dim=-1))

        return x_recon,a_pred,zx,pred_zx_prime

    def inference(self, x_ref):

        zx = self.x_encoder(x_ref)
        za = self.latent_action(zx)

        vq_za, _ = self.vq(za)

        x_current = x_ref[:, 1:-1, :]
        a_pred = self.a_decoder(torch.cat((vq_za, x_current), dim=-1))

        return a_pred


class Single_Koopman_AutoEncoder_Cotrain(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dims, mask_dim,latent_dim,block_dim=None,backbone='mlp',target_k=-1):
        super(Single_Koopman_AutoEncoder_Cotrain, self).__init__()

        self.backbone = backbone
        self.block_dim = block_dim
        if block_dim:
            assert latent_dim%block_dim==0
            print(f'init Koopman_AutoEncoder_Cotrain with self.backbone:{self.backbone}, block_dim:{block_dim}, latent_dim:{latent_dim}, target_k:{target_k}')

        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) > 2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'[KPM Single Cotrain] init input dim:{input_dim}')

        if self.backbone=='mlp':
            self.x_encoder = MLPNetwork(input_dim, hidden_dims, latent_dim, factorize=False)
            self.x_decoder = MLPNetwork(latent_dim, hidden_dims, input_dim, factorize=False)

        elif self.backbone=='lstm':
            self.x_encoder = LSTMNetwork(input_dim, hidden_dims, latent_dim,bidirectional=False)
            self.x_decoder = LSTMNetwork(latent_dim, hidden_dims, input_dim,bidirectional=False)


        if block_dim:
            num_block = latent_dim//block_dim
            block_dim_list = list(np.ones(num_block,dtype=int)*block_dim)
            # self.koopman_hat = nn.ParameterList([nn.Parameter(torch.randn(d, d)) for d in block_dim_list])
            self.koopman = nn.ParameterList([nn.Parameter(torch.randn(d, d)) for d in block_dim_list])
            print(torch.block_diag(*self.koopman_hat))
        else:
            # self.koopman_hat = nn.Parameter(torch.randn(latent_dim))
            self.koopman = nn.Parameter(torch.randn(latent_dim))

        self.z_act = MLPNetwork(3 * input_dim, [256, 256], latent_dim, factorize=False)
        self.act_decoder = nn.Linear(latent_dim,action_dim)
        self.k = target_k
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obs_forward(self,x,training=True):

        if training:
            zx = self.x_encoder(x)
            x_recon = self.x_decoder(zx)

            return zx,x_recon

        else:
            return self.x_encoder(x)

    def get_target(self,zx):

        if self.k==-1:
            zx_last = zx[:, -1, :].unsqueeze(1)
            zx_target = zx_last.expand(-1, zx[:,2:,:].shape[1], -1)
        else:
            # zx_target: for each time step t, zx_target is zx_{t+k}
            t_max = zx.shape[1] - 1
            indices = torch.arange(zx.shape[1], device=zx.device).unsqueeze(0).unsqueeze(-1)
            indices = indices.expand(zx.shape[0], -1, zx.shape[2]) + self.k
            indices = torch.clamp(indices, max=t_max)
            zx_target = torch.gather(zx.clone(), 1, indices)[:,:-2,:]

        return zx_target

    def get_z_act(self,x):

        x_target = self.get_target(x)
        idm_za = self.z_act(torch.cat((x[:, 0:-2, :], x[:, 1:-1, :], x_target), dim=-1))
        return idm_za

    def forward(self,x):

        # full length zx, x_recon
        zx, x_recon = self.obs_forward(x)

        # future length, linear dynamics pred
        zx_current = zx[:, 1:-1, :]
        zx_prime = zx[:,2:,:] # z_t=2

        idm_za = self.get_z_act(x)

        pred_zx_prime = zx_current * self.koopman + idm_za
        a_pred = self.act_decoder(idm_za)


        return x_recon, a_pred, zx_prime, pred_zx_prime,zx

    def inference(self,x):

        idm_za = self.get_z_act(x)
        a_pred = self.act_decoder(idm_za)

        return a_pred




class KPM_Nonlinear_Model(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dims, mask_dim,latent_dim,block_dim=None,backbone='mlp',target_k=-1):
        super(KPM_Nonlinear_Model, self).__init__()

        self.backbone = backbone
        self.block_dim = block_dim
        if block_dim:
            assert latent_dim%block_dim==0
            print(f'init KPM_Nonlinear_Model with self.backbone:{self.backbone}, block_dim:{block_dim}, latent_dim:{latent_dim}, target_k:{target_k}')

        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) > 2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements

        print(f'[KPM_Nonlinear_Model] init input dim:{input_dim}, latent_dim:{latent_dim}')

        if self.backbone=='mlp':
            self.x_encoder = MLPNetwork(input_dim, hidden_dims, latent_dim, factorize=False)
            self.x_decoder = MLPNetwork(latent_dim, hidden_dims, input_dim, factorize=False)

        elif self.backbone=='lstm':
            self.x_encoder = LSTMNetwork(input_dim, hidden_dims, latent_dim,bidirectional=False)
            self.x_decoder = LSTMNetwork(latent_dim, hidden_dims, input_dim,bidirectional=False)


        self.dynamics = LSTMNetwork(2*latent_dim,[256],latent_dim)

        self.z_act = MLPNetwork(3 * input_dim, [256, 256], latent_dim, factorize=False)
        # self.act_decoder = nn.Linear(latent_dim,action_dim)
        self.act_decoder = LSTMNetwork(latent_dim,[256], action_dim, bidirectional=False)

        self.k = target_k
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def obs_forward(self,x,training=True):

        if training:
            zx = self.x_encoder(x)
            x_recon = self.x_decoder(zx)

            return zx,x_recon

        else:
            return self.x_encoder(x)

    def get_target(self,zx):

        if self.k==-1:
            zx_last = zx[:, -1, :].unsqueeze(1)
            zx_target = zx_last.expand(-1, zx[:,2:,:].shape[1], -1)
        else:
            # zx_target: for each time step t, zx_target is zx_{t+k}
            t_max = zx.shape[1] - 1
            indices = torch.arange(zx.shape[1], device=zx.device).unsqueeze(0).unsqueeze(-1)
            indices = indices.expand(zx.shape[0], -1, zx.shape[2]) + self.k
            indices = torch.clamp(indices, max=t_max)
            zx_target = torch.gather(zx.clone(), 1, indices)[:,:-2,:]

        return zx_target

    def get_z_act(self,x):

        x_target = self.get_target(x)
        idm_za = self.z_act(torch.cat((x[:, 0:-2, :], x[:, 1:-1, :], x_target), dim=-1))
        return idm_za

    def forward(self,x):

        # full length zx, x_recon
        zx, x_recon = self.obs_forward(x)

        # future length, linear dynamics pred
        zx_current = zx[:, 1:-1, :]
        zx_prime = zx[:,2:,:] # z_t=2

        idm_za = self.get_z_act(x)

        pred_zx_prime = self.dynamics(torch.cat((zx_current,idm_za),dim=-1))
        a_pred = self.act_decoder(idm_za)


        return x_recon, a_pred, zx_prime, pred_zx_prime,zx

    def inference(self,x):

        idm_za = self.get_z_act(x)
        a_pred = self.act_decoder(idm_za)

        return a_pred




class Single_Koopman_AutoEncoder_Pretrain(nn.Module):

    def __init__(self, input_dim, action_dim, hidden_dims, mask_dim,latent_dim,block_dim=None,backbone='mlp',target_k=-1):
        super(Single_Koopman_AutoEncoder_Pretrain, self).__init__()

        self.backbone = backbone
        self.block_dim = block_dim
        if block_dim:
            assert latent_dim % block_dim == 0
            print(
                f'init Sinle_Koopman_AutoEncoder_Pretrain with self.backbone:{self.backbone}, block_dim:{block_dim}, latent_dim:{latent_dim}, target_k:{target_k}')

        if len(mask_dim) == 1:
            input_dim = input_dim - mask_dim[0]
        elif len(mask_dim) == 2:
            input_dim = input_dim - (mask_dim[1] - mask_dim[0])
        elif len(mask_dim) > 2:
            masked_elements = 0
            for i in range(0, len(mask_dim), 2):
                start = mask_dim[i]
                end = mask_dim[i + 1]
                masked_elements += (end - start)
            input_dim = input_dim - masked_elements
        print(f'init input dim:{input_dim}')

        if self.backbone=='mlp':
            self.x_encoder = MLPNetwork(input_dim, hidden_dims, latent_dim, factorize=False)
            self.x_decoder = MLPNetwork(latent_dim, hidden_dims, input_dim, factorize=False)

        elif self.backbone=='lstm':
            self.x_encoder = LSTMNetwork(input_dim, hidden_dims, latent_dim,bidirectional=False)
            self.x_decoder = LSTMNetwork(latent_dim, hidden_dims, input_dim,bidirectional=False)

        self.koopman = nn.Parameter(torch.randn(latent_dim))

        self.z_act = MLPNetwork(3 * input_dim, [256, 256], latent_dim, factorize=False)
        self.act_decoder = nn.Linear(latent_dim,action_dim)
        self.k = target_k
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze_pretrain_parameters(self):
        for name, param in self.named_parameters():
            if 'act_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print('Parameters frozen except for koopman and act_decoder')


    def obs_forward(self,x,training=True):

        if training:
            zx = self.x_encoder(x)
            x_recon = self.x_decoder(zx)

            return zx,x_recon

        else:
            return self.x_encoder(x)

    def get_target(self,zx):

        if self.k==-1:
            zx_last = zx[:, -1, :].unsqueeze(1)
            zx_target = zx_last.expand(-1, zx[:,2:,:].shape[1], -1)
        else:
            # zx_target: for each time step t, zx_target is zx_{t+k}
            t_max = zx.shape[1] - 1
            indices = torch.arange(zx.shape[1], device=zx.device).unsqueeze(0).unsqueeze(-1)
            indices = indices.expand(zx.shape[0], -1, zx.shape[2]) + self.k
            indices = torch.clamp(indices, max=t_max)
            zx_target = torch.gather(zx.clone(), 1, indices)[:,:-2,:]

        return zx_target

    def get_z_act(self,x):

        x_target = self.get_target(x)
        idm_za = self.z_act(torch.cat((x[:, 0:-2, :], x[:, 1:-1, :], x_target), dim=-1))
        return idm_za

    def forward(self,x):

        # full length zx, x_recon
        zx, x_recon = self.obs_forward(x)

        # future length, linear dynamics pred
        zx_current = zx[:, 1:-1, :]
        zx_prime = zx[:,2:,:] # z_t=2

        idm_za = self.get_z_act(x)

        pred_zx_prime = zx_current * self.koopman + idm_za

        return x_recon, zx_prime, pred_zx_prime,zx

    def finetune(self,x):

        # full length zx, x_recon
        zx, x_recon = self.obs_forward(x)

        # future length, linear dynamics pred
        zx_current = zx[:, 1:-1, :]
        zx_prime = zx[:, 2:, :]  # z_t=2

        idm_za = self.get_z_act(x)

        pred_zx_prime = zx_current * self.koopman + idm_za

        pred_a = self.act_decoder(idm_za)

        return pred_a, x_recon, zx_prime, pred_zx_prime,zx

    def inference(self,x):
        idm_za = self.get_z_act(x)
        pred_a = self.act_decoder(idm_za)

        return pred_a



if __name__ == "__main__":


    # net = VariationalAutoEncoder(input_dim=8,action_dim=3,hidden_dims=[256],latent_dim=4, backbone='lstm')
    # input = torch.randn(256, 16, 5)
    # x_recon, pred_act, z_mu, z_var = net.forward(input)
    # print(x_recon.shape, pred_act.shape, z_var.shape, z_mu.shape)
    # a_pred = net.inference(input)
    # print(a_pred.shape)

    net = VariationalAutoEncoder(input_dim=8,action_dim=3,hidden_dims=[256],latent_dim=4, backbone='lstm')
    input1 = torch.randn(256, 16, 5)
    input2 = torch.randn(256, 16, 3)
    out = net.forward(input1,input2)
    for elem in out:
        print(elem.size())
    a_pred = net.inference(input1)
    print(a_pred.shape)


    # net = AutoEncoderFSQ(input_dim=8,action_dim=3,hidden_dims=[256],latent_dim=32, backbone='lstm')
    # input = torch.randn(256, 16, 5)
    # pred_x, pred_act = net.forward(input)
    # print(pred_x.shape, pred_act.shape)
    # a_pred = net.inference(input)
    # print(a_pred.shape)


    # net = LatentDynamicsFSQ(input_dim=8,action_dim=3,hidden_dims=[256],latent_dim=8, backbone='mlp')
    # input = torch.randn(256,16,5)
    # x_recon,a_pred,zx_prime,pred_zx_prime = net.forward(input)
    # print(x_recon.shape, a_pred.shape,zx_prime.shape, pred_zx_prime.shape)
    # a_pred = net.inference(input)
    # print(a_pred.shape)


    # net = Koopman_AutoEncoder_Cotrain(input_dim=8,action_dim=2,hidden_dims=[256],latent_dim=256, backbone='lstm')
    # input = torch.randn(256, 16, 6)
    # out = net.forward(input)
    # for elem in out:
    #     print(elem.size())
    # a_pred = net.inference(input)
    # print(a_pred.shape)




