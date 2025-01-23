import torch
import numpy as np
import torch.nn.functional as F
from geoopt import linalg

class SPDTSW:
    """
    Class for computing Tree Sliced Wasserstein distance on SPD matrices.
    """

    def __init__(
        self,
        d,
        ntrees=1000,
        nlines=5,
        p=2,
        delta=2,
        mass_division="distance_based",
        device="cuda",
        dtype=torch.float64,
        random_state=123456,
    ):
        self.d = d
        self.ntrees = ntrees
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.mass_division = mass_division
        self.device = device
        self.dtype = dtype
        self.random_state = random_state

        # Initialize SPDSW projection matrices
        self.generate_projections()
        
    def grad_proj(self, x, v):
        #print(v.shape)
        return v.reshape(v.shape[0], v.shape[1], 1, self.d * self.d)

    def generate_projections(self):
        # Generate random theta
        theta = torch.randn((self.ntrees, self.nlines, self.d), dtype=self.dtype, device=self.device)
        theta = F.normalize(theta, p=2, dim=-1)

        # Construct diagonal matrices D
        D = theta[:, :, None] * torch.eye(self.d, device=self.device, dtype=self.dtype)

        # Generate random orthogonal matrices
        Z = torch.randn((self.ntrees, self.nlines, self.d, self.d), dtype=self.dtype, device=self.device)
        Q, R = torch.linalg.qr(Z)
        lambd = torch.diagonal(R, dim1=-2, dim2=-1)
        lambd = lambd / torch.abs(lambd)
        P = lambd[:, :, None] * Q

        # Compute projection matrices A
        # A is symmetric matrix with frob norm 1
        self.A = torch.matmul(P, torch.matmul(D, P.transpose(-2, -1))) # (ntrees, nlines, d, d)
        
        random_matrix = torch.randn((self.ntrees, self.d, self.d), dtype=self.dtype, device=self.device) * 0.000001

        # R is symmetric matrix
        self.R = 0.5 * (random_matrix + random_matrix.transpose(-2, -1)) # (ntrees, d, d)


    def get_mass_and_coordinate(self, Xs, Xt):
        """
        Compute combined coordinates and mass for SPD matrices.
        Args:
            Xs: Source SPD matrices, shape (n, d, d)
            Xt: Target SPD matrices, shape (m, d, d)
        Returns:
            combined_axis_coordinate: Tensor of shape (ntrees, nlines, n + m)
            mass: Tensor of shape (ntrees, nlines, n + m)
        """
        n, _, _ = Xs.shape
        m, _, _ = Xt.shape
        
        # Busemann Coordinates
        log_Xs = linalg.sym_logm(Xs).unsqueeze(0) - self.R.unsqueeze(1) # (t, n, d, d)
        log_Xt = linalg.sym_logm(Xt).unsqueeze(0) - self.R.unsqueeze(1) # (t, m, d, d)

        # Project SPD matrices
        trace_A_log_Xs = torch.einsum('tlij,tnji->tln', self.A, log_Xs)
        trace_A_log_Xt = torch.einsum('tlij,tmji->tlm', self.A, log_Xt)
        
        combined_axis_coordinate = torch.cat((trace_A_log_Xs, trace_A_log_Xt), dim=-1)
        if self.mass_division == "uniform":
            mass_Xs = torch.ones_like(trace_A_log_Xs) / (n * self.nlines)
            mass_Xt = torch.ones_like(trace_A_log_Xt) / (m * self.nlines)
        elif self.mass_division == "distance_based":
            # Compute log of the projected matrices directly
            log_P_GA_Xs = torch.einsum('tln,tlij->tlnij', trace_A_log_Xs, self.A)  # (ntrees, nlines, n, d, d)
            log_P_GA_Xt = torch.einsum('tlm,tlij->tlmij', trace_A_log_Xt, self.A)  # (ntrees, nlines, m, d, d)

            # Compute Frobenius norm of log_M - log(P_GA_M) in parallel
            dist_Xs = torch.linalg.norm(
                log_Xs.unsqueeze(1) - log_P_GA_Xs, dim=(-2, -1), ord="fro"
            )  # (ntrees, nlines, n)
            dist_Xt = torch.linalg.norm(
                log_Xt.unsqueeze(1) - log_P_GA_Xt, dim=(-2, -1), ord="fro"
            )  # (ntrees, nlines, m)
            
            # Compute mass using softmax over distances
            weight_Xs = -self.delta * dist_Xs
            mass_Xs = torch.softmax(weight_Xs, dim=-2) / n  # (ntrees, nlines, n)

            weight_Xt = -self.delta * dist_Xt
            mass_Xt = torch.softmax(weight_Xt, dim=-2) / m  # (ntrees, nlines, m)


        mass = torch.cat((mass_Xs, -mass_Xt), dim=-1)
        #print("Mass_Xs shape: ", mass_Xs.shape)
        return combined_axis_coordinate, mass

    def tw_concurrent_lines(self, mass_XY, combined_axis_coordinate):
        """
        Compute Tree Wasserstein distance.
        Args:
            mass_XY: Tensor of shape (ntrees, nlines, n + m)
            combined_axis_coordinate: Tensor of shape (ntrees, nlines, n + m)
        Returns:
            tw: Scalar distance
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # Generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
        sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]

        # Compute edge length
        root = torch.zeros(num_trees, num_lines, 1, device=self.device)
        root_indices = torch.searchsorted(coord_sorted, root)
        coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device, dtype=self.dtype)
        edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
        edge_mask.scatter_(2, root_indices, False)
        coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
        edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]

        # Compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * edge_length
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1, -2])
        
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, subtract_mass_sum, edge_length

    def spdtsw(self, Xs, Xt):
        """
        Compute Tree Sliced Wasserstein distance for SPD matrices.
        Args:
            Xs: Tensor of shape (n, d, d), SPD matrices from source
            Xt: Tensor of shape (m, d, d), SPD matrices from target
        Returns:
            Scalar: Tree Sliced Wasserstein distance
        """
        combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(Xs, Xt)
        nabla_proj = self.grad_proj(Xs, self.A)

        sub_mass_target_cumsum = self.tw_concurrent_lines(mass_XY, combined_axis_coordinate)[1]
        #print("d_potential[:, :, None] shape", sub_mass_target_cumsum.shape) #10, 10, 4
        #print("nabla_prj shape:", nabla_proj.shape) #10, 4, 
        sub_mass_target_cumsum_mul = (sub_mass_target_cumsum[:, :, :, None] * nabla_proj)
        tw = torch.mean(sub_mass_target_cumsum_mul) ** (1/self.p)
        return tw
