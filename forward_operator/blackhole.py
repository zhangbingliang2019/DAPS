import numpy as np
import torch
from . import register_operator, Operator


@register_operator(name='blackhole_imaging')
class BlackHoleImaging(Operator):
    '''
    Black hole imaging forward model
    NOTE: The model is hard-coded for a single blackhole image reconstruction
    '''

    def __init__(self, root='dataset/blackhole', imsize=64, w1=0, w2=1, w3=1, w4=0.5, device='cuda', sigma=0) -> None:
        super().__init__(sigma)
        import ehtim as eh
        torch.manual_seed(873345)
        # load observations
        A_vis, A_cp, A_camp, obs, im = self.process_obs(root, imsize)
        # sigmas
        self.sigma_amp = torch.tensor(obs.amp['sigma']).unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(device)
        self.sigma_cp = torch.tensor(obs.cphase['sigmacp'] * eh.DEGREE).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).float().to(device)
        self.sigma_camp = torch.tensor(obs.logcamp['sigmaca']).unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(
            device)
        # measurements
        self.y_amp = torch.from_numpy(obs.amp['amp']).unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(device)
        self.y_cp = torch.from_numpy(obs.cphase['cphase'] * eh.DEGREE).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).float().to(device)
        self.y_camp = torch.from_numpy(obs.logcamp['camp']).unsqueeze(0).unsqueeze(0).unsqueeze(-1).float().to(device)
        self.y_flux = torch.tensor(self.estimate_flux(obs)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).float().to(device)
        # forward matrix
        self.A_vis = torch.from_numpy(A_vis).unsqueeze(0).unsqueeze(0).cfloat().to(device)  # [1,1,m,n]
        self.A_cp = torch.from_numpy(A_cp).unsqueeze(1).unsqueeze(1).cfloat().to(device)  # [3,1,1,m,n]
        self.A_camp = torch.from_numpy(A_camp).unsqueeze(1).unsqueeze(1).cfloat().to(device)  # [4,1,1,m,n]
        # naive recon (NOTE:TO BE CORRECTED)
        self.naive_recon = obs.dirtyimage(64, im.fovx()).ivec.reshape(imsize, imsize)
        # params
        self.C = 1
        self.H = imsize
        self.W = imsize
        self.weight_amp = w1 * A_vis.shape[0]
        self.weight_cp = w2 * A_cp.shape[1]
        self.weight_camp = w3 * A_camp.shape[1]
        self.weight_flux = w4

        # import matplotlib.pyplot as plt
        # import ehtplot.color
        # plt.figure()
        # plt.imshow(self.naive_recon, cmap='afmhot_10us')
        # # plt.axis('off')
        # plt.colorbar()
        # plt.savefig('adj.png', dpi=400, bbox_inches='tight')
        # plt.close()
        # exit()

        # xtrue = torch.from_numpy(im.ivec).unsqueeze(0).unsqueeze(0)
        # print(self.chi2_cphase(xtrue, self.y_cp))
        # print(self.chi2_logcamp(xtrue, self.y_camp))
        # exit()

    def __call__(self, x):
        return torch.zeros_like(x)

    def forward_amp(self, x):
        '''
        Args:
            x: input tensor with shape [N, C, H, W]
        Outs:
            vis_amp: output tensor with shape [N, 938]
        '''
        x = x.to(self.A_vis)
        xvec = x.reshape(-1, self.C, self.H * self.W, 1)
        vis = torch.matmul(self.A_vis, xvec)
        vis_amp = torch.abs(vis)
        return vis_amp

    def forward_cp(self, x):
        '''
        Args:
            x: input tensor with shape [N, C, H, W]
        Outs:
            cphase: output tensor with shape [N, 580]
        '''
        x = x.to(self.A_cp)
        xvec = x.reshape(-1, self.C, self.H * self.W, 1)
        i1 = torch.matmul(self.A_cp[0], xvec)
        i2 = torch.matmul(self.A_cp[1], xvec)
        i3 = torch.matmul(self.A_cp[2], xvec)
        cphase = torch.angle(i1 * i2 * i3)
        return cphase

    def forward_logcamp(self, x):
        '''
        Args:
            x: input tensor with shape [N, C, H, W]
        Outs:
            camp: output tensor with shape [N, 580]
        '''
        x = x.to(self.A_camp)
        x_vec = x.reshape(-1, self.C, self.H * self.W, 1)
        a1 = torch.abs(torch.matmul(self.A_camp[0], x_vec))
        a2 = torch.abs(torch.matmul(self.A_camp[1], x_vec))
        a3 = torch.abs(torch.matmul(self.A_camp[2], x_vec))
        a4 = torch.abs(torch.matmul(self.A_camp[3], x_vec))
        camp = torch.log(a1) + torch.log(a2) - torch.log(a3) - torch.log(a4)
        return camp

    def grad(self, x):
        grad_amp = self.chisqgrad_amp(x, self.y_amp)
        grad_cp = self.chisqgrad_cphase(x, self.y_cp)
        # grad_camp = self.chisqgrad_camp(x, self.y_camp)
        grad_camp = self.chisqgrad_logcamp(x, self.y_camp)
        grad_flux = self.chisqgrad_flux(x, self.y_flux)
        grad = self.weight_amp * grad_amp + self.weight_cp * grad_cp + self.weight_camp * grad_camp + self.weight_flux * grad_flux
        return grad

    def chisqgrad_amp(self, x, y_amp):
        """The gradient of the amplitude chi-squared"""
        x = x.to(self.A_vis)
        x_vec = x.reshape(-1, self.C, self.H * self.W, 1)
        i1 = torch.matmul(self.A_vis, x_vec)
        amp_samples = torch.abs(i1)
        pp = ((y_amp - amp_samples) * amp_samples) / (self.sigma_amp ** 2) / i1
        out = (-2.0 / y_amp.shape[2]) * torch.real(torch.matmul(pp.mT, self.A_vis))
        return out.reshape(-1, self.C, self.H, self.W)

    def chisqgrad_cphase(self, x, y_cp):
        """The gradient of the closure phase chi-squared"""
        x = x.to(self.A_cp)
        x_vec = x.reshape(-1, self.C, self.H * self.W, 1)
        i1 = torch.matmul(self.A_cp[0], x_vec)
        i2 = torch.matmul(self.A_cp[1], x_vec)
        i3 = torch.matmul(self.A_cp[2], x_vec)
        cphase_samples = torch.angle(i1 * i2 * i3)

        pref = torch.sin(y_cp - cphase_samples) / (self.sigma_cp ** 2)
        pt1 = pref / i1
        pt2 = pref / i2
        pt3 = pref / i3
        out = torch.matmul(pt1.mT, self.A_cp[0]) + torch.matmul(pt2.mT, self.A_cp[1]) + torch.matmul(pt3.mT,
                                                                                                     self.A_cp[2])
        out = (-2.0 / y_cp.shape[2]) * torch.imag(out)
        return out.reshape(-1, self.C, self.H, self.W)

    def chisqgrad_logcamp(self, x, y_camp):
        """The gradient of the Log closure amplitude chi-squared"""
        x = x.to(self.A_camp)
        x_vec = x.reshape(-1, self.C, self.H * self.W, 1)
        i1 = torch.matmul(self.A_camp[0], x_vec)
        i2 = torch.matmul(self.A_camp[1], x_vec)
        i3 = torch.matmul(self.A_camp[2], x_vec)
        i4 = torch.matmul(self.A_camp[3], x_vec)
        log_clamp_samples = (torch.log(torch.abs(i1)) +
                             torch.log(torch.abs(i2)) -
                             torch.log(torch.abs(i3)) -
                             torch.log(torch.abs(i4)))

        pp = (y_camp - log_clamp_samples) / (self.sigma_camp ** 2)
        pt1 = pp / i1
        pt2 = pp / i2
        pt3 = -pp / i3
        pt4 = -pp / i4
        out = (
                torch.matmul(pt1.mT, self.A_camp[0]) +
                torch.matmul(pt2.mT, self.A_camp[1]) +
                torch.matmul(pt3.mT, self.A_camp[2]) +
                torch.matmul(pt4.mT, self.A_camp[3])
        )
        out = (-2.0 / y_camp.shape[2]) * torch.real(out)
        return out.reshape(-1, self.C, self.H, self.W)

    def chisqgrad_flux(self, x, y_flux):
        x_vec = x.reshape(-1, self.C, self.H * self.W, 1)
        res = torch.sum(x_vec, dim=(1, 2, 3), keepdim=True) - y_flux
        out = torch.ones_like(x_vec) * res
        return out.reshape(-1, self.C, self.H, self.W)

    def eval(self, x, y):
        chi2_amp_val = self.chi2_amp(x, y[0])
        chi2_cp_val = self.chi2_cphase(x, y[1])
        # chi2_camp_val = self.chi2_camp(x, y[2])
        chi2_camp_val = self.chi2_logcamp(x, y[2])
        data_fit = self.weight_amp * chi2_amp_val + self.weight_cp * chi2_cp_val + self.weight_camp * chi2_camp_val
        return data_fit, chi2_amp_val, chi2_cp_val, chi2_camp_val

    def chi2_amp(self, x, y_amp):
        amp_pred = self.forward_amp(x)
        residual = y_amp - amp_pred
        return torch.mean(torch.square(residual / self.sigma_amp), dim=(1, 2, 3))

    def chi2_cphase(self, x, y_cphase):
        cphase_pred = self.forward_cp(x)
        angle_residual = y_cphase - cphase_pred
        return 2. * torch.mean((1 - torch.cos(angle_residual)) / torch.square(self.sigma_cp), dim=(1, 2, 3))

    def chi2_logcamp(self, x, y_camp):
        y_camp_pred = self.forward_logcamp(x)
        return torch.mean(torch.abs((y_camp - y_camp_pred) / self.sigma_camp) ** 2, dim=(1, 2, 3))

    @staticmethod
    def process_obs(
            root,
            imsize,
    ):
        obsfile = root + '/' + 'obs.uvfits'
        gtfile = root + '/' + 'gt.fits'
        # load measurements generated by Berthy
        obs = eh.obsdata.load_uvfits(obsfile)
        im = eh.image.load_fits(gtfile)
        im = im.regrid_image(im.fovx(), imsize)
        # Rescale image
        multiplier = round(1 / im.ivec.max())
        im.ivec = im.ivec * multiplier
        # Scale visibilities and visibility sigmas.
        obs.data['vis'] *= multiplier
        obs.data['sigma'] *= multiplier
        # Compute visibility amplitudes and closure phases
        obs.add_amp(debias=True)
        obs.add_cphase(count='min')
        # obs.add_camp(count='min')
        obs.add_logcamp(count='min')
        # Get forward model for complex visibilities.
        _, _, A_vis = eh.imaging.imager_utils.chisqdata_vis(obs, im, mask=[])
        # Get forward model for closure phases.
        _, _, A_cp = eh.imaging.imager_utils.chisqdata_cphase(obs, im, mask=[])
        # Get forward model for closure amplitudes.
        _, _, A_camp = eh.imaging.imager_utils.chisqdata_logcamp(obs, im, mask=[])
        return A_vis, np.stack(A_cp, axis=0), np.stack(A_camp, axis=0), obs, im

    @staticmethod
    def estimate_flux(obs):
        # estimate the total flux from the observation
        data = obs.unpack_bl('ALMA', 'APEX', 'amp')
        amp_list = []
        for pair in data:
            amp = pair[0][1]
            amp_list.append(amp)
        flux = np.median(amp_list)
        return flux

