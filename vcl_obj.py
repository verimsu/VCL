import torch.nn as nn
import torch.distributions as tdist

class VCLObjective(nn.Module):
    def __init__(self):
        super(VCLObjective, self).__init__()

    def jsd_gaussian(self, mu_p, mu_q, log_var_p, log_var_q):
        std_p = (log_var_p / 2).exp()
        std_q = (log_var_q / 2).exp()

        # Create Gaussian distributions using mean and std
        p_dist = tdist.Normal(mu_p, std_p)
        q_dist = tdist.Normal(mu_q, std_q)

        # Calculate M as the average of the two distributions
        m_mean = (mu_p + mu_q) / 2
        m_std = (((log_var_p.exp() + log_var_q.exp()) / 2).log() / 2).exp()
        m_dist = tdist.Normal(m_mean, m_std)

        # Calculate KL divergence between each distribution and M
        kl_p_m = tdist.kl_divergence(p_dist, m_dist)
        kl_q_m = tdist.kl_divergence(q_dist, m_dist)

        # Calculate JSD as the average of the KL divergences
        jsd = 0.5 * (kl_p_m + kl_q_m)
        
        return jsd
    
    def forward(self, mu_p, mu_q, log_var_p, log_var_q):
        jsd = self.jsd_gaussian(mu_p, mu_q, log_var_p, log_var_q).mean()
        
        # Calculate the normalization term
        norm_p = -0.5 * (1 + log_var_p - mu_p.pow(2) - log_var_p.exp()).mean()
        norm_q = -0.5 * (1 + log_var_q - mu_q.pow(2) - log_var_q.exp()).mean()
        norm = norm_p + norm_q
        
        return jsd + norm