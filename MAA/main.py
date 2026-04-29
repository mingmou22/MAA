import math
import argparse
import random
from collections import deque

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from matplotlib.colors import TwoSlopeNorm


def rgb_to_hsv(img):
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc, _ = torch.max(img, dim=1)
    minc, _ = torch.min(img, dim=1)
    v = maxc

    deltac = maxc - minc + 1e-6
    s = deltac / (maxc + 1e-6)

    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    h = torch.zeros_like(maxc)
    h[r == maxc] = (bc - gc)[r == maxc]
    h[g == maxc] = 2.0 + (rc - bc)[g == maxc]
    h[b == maxc] = 4.0 + (gc - rc)[b == maxc]
    h = (h / 6.0) % 1.0
    return h, s, v


def hsv_to_rgb(h, s, v):
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.long() % 6

    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    out = torch.zeros(h.size(0), 3, h.size(1), h.size(2), device=h.device)

    idx = (i == 0)
    out[:, 0] = torch.where(idx, v, out[:, 0])
    out[:, 1] = torch.where(idx, t, out[:, 1])
    out[:, 2] = torch.where(idx, p, out[:, 2])

    idx = (i == 1)
    out[:, 0] = torch.where(idx, q, out[:, 0])
    out[:, 1] = torch.where(idx, v, out[:, 1])
    out[:, 2] = torch.where(idx, p, out[:, 2])

    idx = (i == 2)
    out[:, 0] = torch.where(idx, p, out[:, 0])
    out[:, 1] = torch.where(idx, v, out[:, 1])
    out[:, 2] = torch.where(idx, t, out[:, 2])

    idx = (i == 3)
    out[:, 0] = torch.where(idx, p, out[:, 0])
    out[:, 1] = torch.where(idx, q, out[:, 1])
    out[:, 2] = torch.where(idx, v, out[:, 2])

    idx = (i == 4)
    out[:, 0] = torch.where(idx, t, out[:, 0])
    out[:, 1] = torch.where(idx, p, out[:, 1])
    out[:, 2] = torch.where(idx, v, out[:, 2])

    idx = (i == 5)
    out[:, 0] = torch.where(idx, v, out[:, 0])
    out[:, 1] = torch.where(idx, p, out[:, 1])
    out[:, 2] = torch.where(idx, q, out[:, 2])

    return out


class MaskRCNNSegmenter(nn.Module):
    def __init__(self, device):
        super().__init__()
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img_bchw, score_thr=0.7, max_instances=3, fallback="all"):
        device = img_bchw.device
        B, _, H, W = img_bchw.shape
        out_masks = []
        for b in range(B):
            x = img_bchw[b]
            preds = self.model([x])[0]

            if ("masks" not in preds) or preds["masks"].numel() == 0:
                out_masks.append(torch.ones((H, W), device=device) if fallback == "all"
                                 else torch.zeros((H, W), device=device))
                continue

            scores = preds["scores"]
            keep = scores >= score_thr
            if keep.sum().item() == 0:
                out_masks.append(torch.ones((H, W), device=device) if fallback == "all"
                                 else torch.zeros((H, W), device=device))
                continue

            idxs = torch.nonzero(keep).squeeze(1)
            idxs = idxs[torch.argsort(scores[idxs], descending=True)]
            idxs = idxs[:max_instances]

            masks = preds["masks"][idxs, 0]  # (M,H,W) prob
            union_prob = masks.max(dim=0).values  # soft union
            out_masks.append(union_prob)

        return torch.stack(out_masks, dim=0)


def mask_to_bbox(mask_hw, thr=0.3, pad=8):
    H, W = mask_hw.shape
    ys, xs = torch.where(mask_hw > thr)
    if ys.numel() == 0:
        return (0, H, 0, W)
    y0 = int(max(0, ys.min().item() - pad))
    y1 = int(min(H, ys.max().item() + 1 + pad))
    x0 = int(max(0, xs.min().item() - pad))
    x1 = int(min(W, xs.max().item() + 1 + pad))
    return (y0, y1, x0, x1)


def crop_resize_2d_batch(x_bhw, bboxes, out_hw=(128, 128)):
    B, H, W = x_bhw.shape
    out = []
    for b in range(B):
        y0, y1, x0, x1 = bboxes[b]
        crop = x_bhw[b:b+1, y0:y1, x0:x1]  # (1,h,w)
        crop = crop.unsqueeze(1)           # (1,1,h,w)
        crop = F.interpolate(crop, size=out_hw, mode="bilinear", align_corners=False)
        out.append(crop.squeeze(1))        # (1,outH,outW)
    return torch.cat(out, dim=0)


def betweenness_centrality_unweighted(adj: torch.Tensor) -> torch.Tensor:
    A = (adj.detach().cpu() > 0).numpy()
    N = A.shape[0]
    neighbors = []
    for i in range(N):
        nbrs = np.where(A[i])[0].tolist()
        nbrs = [j for j in nbrs if j != i]
        neighbors.append(nbrs)

    bc = np.zeros(N, dtype=np.float64)

    for s in range(N):
        stack = []
        P = [[] for _ in range(N)]
        sigma = np.zeros(N, dtype=np.float64)
        sigma[s] = 1.0
        dist = -np.ones(N, dtype=np.int32)
        dist[s] = 0

        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta = np.zeros(N, dtype=np.float64)
        while stack:
            w = stack.pop()
            for v in P[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]

    bc *= 0.5
    if N > 2:
        bc *= (2.0 / ((N - 1) * (N - 2)))

    return torch.tensor(bc, dtype=torch.float32)


def centrality_weights(adj: torch.Tensor) -> torch.Tensor:
    bc = betweenness_centrality_unweighted(adj).to(adj.device)
    w = (bc - bc.min()) / (bc.max() - bc.min() + 1e-8)
    return w.unsqueeze(1)


class ChebDiffusion(nn.Module):
    def __init__(self, K=3):
        super().__init__()
        self.K = K
        self.register_buffer("theta", torch.tensor([1.0 / (k + 1) for k in range(K + 1)]))

    def scaled_laplacian(self, adj):
        N = adj.size(0)
        I = torch.eye(N, device=adj.device)
        deg = adj.sum(1)
        D_inv_sqrt = torch.diag(torch.pow(deg + 1e-8, -0.5))
        L = I - D_inv_sqrt @ adj @ D_inv_sqrt
        L_tilde = L - I
        return L_tilde

    def cheb_polys(self, L_tilde):
        N = L_tilde.size(0)
        T0 = torch.eye(N, device=L_tilde.device)
        if self.K == 0:
            return [T0]
        T1 = L_tilde
        Ts = [T0, T1]
        for _ in range(2, self.K + 1):
            Ts.append(2 * L_tilde @ Ts[-1] - Ts[-2])
        return Ts

    def forward(self, H0, adj):
        L_tilde = self.scaled_laplacian(adj)
        Ts = self.cheb_polys(L_tilde)
        out = 0
        for k, Tk in enumerate(Ts):
            out = out + self.theta[k] * (Tk @ H0)
        return out


class FractalHOperator(nn.Module):
    def forward(self, H):
        return F.avg_pool2d(H.unsqueeze(1), 3, 1, 1).squeeze(1)


class VGGStyle(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.DEFAULT
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features

        self.layers = nn.Sequential(*list(vgg)[:16])
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def gram(x):
        B, C, H, W = x.shape
        f = x.view(B, C, H * W)
        return torch.bmm(f, f.transpose(1, 2)) / (C * H * W + 1e-8)

    def forward(self, x):
        return self.layers(x)


def img_to_nodes_bchw(x_bchw, patch, stride):
   
    x_bchw: (B,1,H,W)
    return nodes: (B,N,1), (nH,nW)
    """
    B, C, H, W = x_bchw.shape
    assert C == 1
    unfold = torch.nn.Unfold(kernel_size=patch, stride=stride)
    patches = unfold(x_bchw)  # (B, patch*patch, N)

    nodes = patches.mean(dim=1, keepdim=True)  # (B,1,N)
    nodes = nodes.transpose(1, 2)              # (B,N,1)

    nH = (H - patch) // stride + 1
    nW = (W - patch) // stride + 1
    return nodes, (nH, nW)


def nodes_to_img_bchw(nodes_bn1, H, W, patch, stride):
   
    nodes_bn1: (B,N,1)
    return: (B,1,H,W) via overlap-add average
 
    B, N, _ = nodes_bn1.shape
    fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch, stride=stride)

    vals = nodes_bn1.transpose(1, 2)          
    vals = vals.repeat(1, patch * patch, 1)  
    out = fold(vals)                          

    
    ones = torch.ones((B, 1, H, W), device=nodes_bn1.device, dtype=nodes_bn1.dtype)
    unfold = torch.nn.Unfold(kernel_size=patch, stride=stride)
    denom = fold(unfold(ones))                
    out = out / (denom + 1e-8)
    return out


def hue_embed(h):
    ang = 2 * math.pi * h
    return torch.cos(ang), torch.sin(ang)


def graph_smooth_field(field_bhw, adj, diffuser, patch, stride):
    
    
    B, H, W = field_bhw.shape
    nodes, _ = img_to_nodes_bchw(field_bhw.unsqueeze(1), patch=patch, stride=stride)  

    ys = []
    for b in range(B):
        ys.append(diffuser(nodes[b], adj))  # (N,1)
    y = torch.stack(ys, dim=0)  # (B,N,1)

    smoothed = nodes_to_img_bchw(y, H, W, patch=patch, stride=stride).squeeze(1)  
    return smoothed


def qnodes_to_qmap(Q_nodes_n1, H, W, patch, stride, B):
    
    Q_nodes_n1: (N,1) -> (B,H,W) continuous map via fold averaging
    
    q = Q_nodes_n1.unsqueeze(0).repeat(B, 1, 1)  # (B,N,1)
    qmap = nodes_to_img_bchw(q, H, W, patch=patch, stride=stride).squeeze(1)  # (B,H,W)
    return qmap



def structured_hsv_target_attack(
    src_img, tar_img,
    adj,
    segmenter: MaskRCNNSegmenter,
    block_hw=(16, 16),
    stride=None,            
    steps=20,
    eps=8/255,
    alpha=None,
    lam=3.0,
    lam_hf=0.1,
    score_thr=0.7,
    max_instances=3,
    crop_hw=(128, 128),
    cheb_K=3,
    embed_dim=16,
    q_floor=0.2,
    use_graph_smooth=True,
):
    device = src_img.device
    B, _, H, W = src_img.shape
    bh, bw = block_hw
    assert bh == bw, "this implementation assumes square patch"
    patch = bh
    if stride is None:
        stride = patch

    nH = (H - patch) // stride + 1
    nW = (W - patch) // stride + 1
    N = nH * nW
    assert adj.shape == (N, N), f"adj shape {adj.shape} but expected {(N,N)} (check patch/stride)"

    if alpha is None:
        alpha = float(eps) / float(steps)

    FH = FractalHOperator().to(device)
    vgg = VGGStyle().to(device)

    
    with torch.no_grad():
        mask_src = segmenter(src_img, score_thr=score_thr, max_instances=max_instances, fallback="all")
        mask_tar = segmenter(tar_img, score_thr=score_thr, max_instances=max_instances, fallback="all")

    bboxes_src = [mask_to_bbox(mask_src[b], thr=0.3, pad=8) for b in range(B)]
    bboxes_tar = [mask_to_bbox(mask_tar[b], thr=0.3, pad=8) for b in range(B)]

   
    w = centrality_weights(adj).to(device)  # (N,1)

    
    diffuser = ChebDiffusion(K=cheb_K).to(device)
    z = torch.randn(N, embed_dim, device=device)
    H0 = w * z
    Q = diffuser(H0, adj)                     
    Q_nodes = Q.norm(dim=1, keepdim=True)      
    Q_nodes = (Q_nodes - Q_nodes.min()) / (Q_nodes.max() - Q_nodes.min() + 1e-8)

    Q_map = qnodes_to_qmap(Q_nodes, H, W, patch=patch, stride=stride, B=B)
    Q_map = (q_floor + (1.0 - q_floor) * Q_map).clamp(0, 1)

    def high_freq_mask(hh, ww, ratio=0.5):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, hh, device=device),
            torch.linspace(-1, 1, ww, device=device),
            indexing='ij'
        )
        rr = torch.sqrt(xx**2 + yy**2)
        return (rr > ratio).float()

    adv = src_img.clone().detach()

    for _ in range(steps):
        adv.requires_grad_(True)

        Hs, Ss, Vs = rgb_to_hsv(adv)
        Ht, St, Vt = rgb_to_hsv(tar_img)

        
        Hs_c = crop_resize_2d_batch(Hs, bboxes_src, crop_hw)
        Ss_c = crop_resize_2d_batch(Ss, bboxes_src, crop_hw)
        Vs_c = crop_resize_2d_batch(Vs, bboxes_src, crop_hw)

        Ht_c = crop_resize_2d_batch(Ht, bboxes_tar, crop_hw)
        St_c = crop_resize_2d_batch(St, bboxes_tar, crop_hw)
        Vt_c = crop_resize_2d_batch(Vt, bboxes_tar, crop_hw)

        ms_c = crop_resize_2d_batch(mask_src, bboxes_src, crop_hw).clamp(0, 1)
        mt_c = crop_resize_2d_batch(mask_tar, bboxes_tar, crop_hw).clamp(0, 1)

        m_u = (ms_c + mt_c).clamp(0, 1)
        m_int = (ms_c * mt_c).clamp(0, 1)
        m_align = m_int if (m_int.sum() >= 10).item() else m_u

        
        Hs_cos, Hs_sin = hue_embed(Hs_c)
        Ht_cos, Ht_sin = hue_embed(Ht_c)

        aHc = FH(Hs_cos); bHc = FH(Ht_cos)
        aHs = FH(Hs_sin); bHs = FH(Ht_sin)

        loss_H = (
            (((aHc - bHc) * m_align) ** 2).sum() +
            (((aHs - bHs) * m_align) ** 2).sum()
        ) / (m_align.sum() + 1e-8)

        gH = torch.autograd.grad(loss_H, Hs, retain_graph=True)[0]

        
        Fs = vgg((Ss_c * m_u).unsqueeze(1).repeat(1, 3, 1, 1))
        Ft = vgg((St_c * m_u).unsqueeze(1).repeat(1, 3, 1, 1))
        loss_S = F.mse_loss(VGGStyle.gram(Fs), VGGStyle.gram(Ft))
        gS = torch.autograd.grad(loss_S, Ss, retain_graph=True)[0]

        
        Vf_s = torch.fft.fft2(Vs_c * m_u)
        Vf_t = torch.fft.fft2(Vt_c * m_u)
        loss_freq = F.mse_loss(torch.abs(Vf_s), torch.abs(Vf_t))

        Mh = high_freq_mask(crop_hw[0], crop_hw[1], ratio=0.5)
        loss_hf = (Mh * torch.abs(Vf_s)).pow(2).mean()
        loss_V = loss_freq + lam_hf * loss_hf
        gV = torch.autograd.grad(loss_V, Vs)[0]

        
        gH = gH / (gH.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)
        gS = gS / (gS.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)
        gV = gV / (gV.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)

        # graph smoothing (supports stride overlap)
        if use_graph_smooth:
            gH = graph_smooth_field(gH, adj, diffuser, patch=patch, stride=stride)
            gS = graph_smooth_field(gS, adj, diffuser, patch=patch, stride=stride)
            gV = graph_smooth_field(gV, adj, diffuser, patch=patch, stride=stride)

            gH = gH / (gH.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)
            gS = gS / (gS.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)
            gV = gV / (gV.abs().mean(dim=(1, 2), keepdim=True) + 1e-8)

        
        M = mask_src
        dH = (Q_map * M) * torch.tanh(lam * gH)
        dS = (Q_map * M) * torch.tanh(lam * gS)
        dV = (Q_map * M) * torch.tanh(lam * gV)

        H_adv = (Hs + alpha * dH).clamp(0, 1)
        S_adv = (Ss + alpha * dS).clamp(0, 1)
        V_adv = (Vs + alpha * dV).clamp(0, 1)

        adv_hsv_rgb = hsv_to_rgb(H_adv, S_adv, V_adv)

        
        delta = torch.clamp(adv_hsv_rgb - src_img, min=-eps, max=eps)
        adv = torch.clamp(src_img + delta, 0, 1).detach()

    perturbation = (adv - src_img).detach()
    return adv, perturbation, (mask_src, mask_tar), Q_map


def load_image(path, size):
    img = Image.open(path).convert('RGB').resize((size, size))
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img.unsqueeze(0)


def to_numpy_img(x):
    return x.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

def _viz_enhance(delta_np, q=0.995, gamma=0.5, min_v=1e-4):
    """
    delta_np: (H,W) raw delta
    return:
      show: (H,W) normalized for display in [-1,1]
      vlim: actual delta magnitude used for scaling (percentile)
    """
    vlim = float(np.quantile(np.abs(delta_np), q))
    vlim = max(vlim, min_v)

    x = np.clip(delta_np / vlim, -1, 1)            
    x = np.sign(x) * (np.abs(x) ** gamma)          
    return x, vlim


def plot_paper_panel_hsv(src_img, tar_img, adv_img, eps_rgb, save_path=None,
                         viz_q=0.995, viz_gamma=0.5):
    src = src_img[0].detach().cpu().permute(1,2,0).numpy()
    tar = tar_img[0].detach().cpu().permute(1,2,0).numpy()
    adv = adv_img[0].detach().cpu().permute(1,2,0).numpy()

    Hs, Ss, Vs = rgb_to_hsv(src_img)
    Ha, Sa, Va = rgb_to_hsv(adv_img)

    # Hue: shortest circular diff in [-0.5, 0.5]
    dH = ((Ha - Hs + 0.5) % 1.0) - 0.5
    dS = (Sa - Ss)
    dV = (Va - Vs)

    dH_np = dH[0].detach().cpu().numpy()
    dS_np = dS[0].detach().cpu().numpy()
    dV_np = dV[0].detach().cpu().numpy()

    
    dH_show, vH = _viz_enhance(dH_np, q=viz_q, gamma=viz_gamma)
    dS_show, vS = _viz_enhance(dS_np, q=viz_q, gamma=viz_gamma)
    dV_show, vV = _viz_enhance(dV_np, q=viz_q, gamma=viz_gamma)

    fig = plt.figure(figsize=(14, 7))

    
    ax1 = plt.subplot(2, 3, 1); ax1.imshow(np.clip(src,0,1)); ax1.axis('off'); ax1.set_title('Source')
    ax2 = plt.subplot(2, 3, 2); ax2.imshow(np.clip(tar,0,1)); ax2.axis('off'); ax2.set_title('Target')
    ax3 = plt.subplot(2, 3, 3); ax3.imshow(np.clip(adv,0,1)); ax3.axis('off'); ax3.set_title('Adversarial')

    
    def draw(ax, show, vlim, title):
        im = ax.imshow(show, cmap='seismic', vmin=-1, vmax=1)
        ax.axis('off')
        ax.set_title(f'{title} (|Δ|@q={viz_q:.3f}={vlim:.4f})')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = [-1, -0.5, 0, 0.5, 1]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f'{t*vlim:.3f}' for t in ticks])  
        return im

    ax4 = plt.subplot(2, 3, 4); draw(ax4, dH_show, vH, 'ΔH')
    ax5 = plt.subplot(2, 3, 5); draw(ax5, dS_show, vS, 'ΔS')
    ax6 = plt.subplot(2, 3, 6); draw(ax6, dV_show, vV, 'ΔV')

    plt.suptitle(f'HSV perturbations (RGB eps={eps_rgb:.5f})', y=1.02, fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def hsv_delta_composite(adv_img, src_img, q=0.995):
    Hs, Ss, Vs = rgb_to_hsv(src_img)
    Ha, Sa, Va = rgb_to_hsv(adv_img)

    # Hue circular diff in [-0.5, 0.5]
    dH = ((Ha - Hs + 0.5) % 1.0) - 0.5
    dS = (Sa - Ss)
    dV = (Va - Vs)

    dH_np = dH[0].detach().cpu().numpy()
    dS_np = dS[0].detach().cpu().numpy()
    dV_np = dV[0].detach().cpu().numpy()


    hlim = float(np.quantile(np.abs(dH_np), q)); hlim = max(hlim, 1e-4)
    slim = float(np.quantile(np.abs(dS_np), q)); slim = max(slim, 1e-4)
    vlim = float(np.quantile(np.abs(dV_np), q)); vlim = max(vlim, 1e-4)

    # Hue: normalize ΔH to [-1,1] then map to [0,1]
    h_norm = np.clip(dH_np / hlim, -1, 1)
    hue = (h_norm * 0.5 + 0.5).astype(np.float32)  # 0..1

    
    strength = np.sqrt((dS_np / slim) ** 2 + (dV_np / vlim) ** 2)
    s_lim2 = float(np.quantile(strength, q)); s_lim2 = max(s_lim2, 1e-6)
    strength = np.clip(strength / s_lim2, 0, 1).astype(np.float32)

    sat = np.ones_like(strength, dtype=np.float32)

    hsv_vis = torch.from_numpy(np.stack([hue, sat, strength], axis=-1)).unsqueeze(0)  # (1,H,W,3)
    hsv_vis = hsv_vis.permute(0, 3, 1, 2).contiguous()  # (1,3,H,W)

    rgb_vis = hsv_to_rgb(hsv_vis[:, 0], hsv_vis[:, 1], hsv_vis[:, 2])[0]  # (3,H,W)
    return rgb_vis.permute(1, 2, 0).clamp(0, 1).cpu().numpy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='2.png')
    parser.add_argument('--target_img', type=str, default='4.jpg')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--stride', type=int, default=None, help='sliding window stride (default=patch_size)')

    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--lam', type=float, default=3.0)
    parser.add_argument('--lam_hf', type=float, default=0.1)

    parser.add_argument('--score_thr', type=float, default=0.7)
    parser.add_argument('--max_instances', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=128)

    parser.add_argument('--cheb_K', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--q_floor', type=float, default=0.2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    src_img = load_image(args.img, args.img_size).to(device)
    tar_img = load_image(args.target_img, args.img_size).to(device)

    B, _, H, W = src_img.shape
    patch = args.patch_size
    stride = args.stride if args.stride is not None else patch

    # sliding-window grid size
    nH = (H - patch) // stride + 1
    nW = (W - patch) // stride + 1
    N = nH * nW

    # 8-neighborhood adjacency on (nH,nW) grid
    adj = torch.zeros(N, N, device=device)

    def idx(i, j): return i * nW + j

    for i in range(nH):
        for j in range(nW):
            u = idx(i, j)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nH and 0 <= nj < nW:
                    v = idx(ni, nj)
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
    adj = adj + torch.eye(N, device=device)

    segmenter = MaskRCNNSegmenter(device)

    adv_img, perturbation, masks, Q_map = structured_hsv_target_attack(
        src_img, tar_img,
        adj=adj,
        segmenter=segmenter,
        block_hw=(patch, patch),
        stride=stride,
        steps=args.n_steps,
        eps=args.eps,
        alpha=args.alpha,
        lam=args.lam,
        lam_hf=args.lam_hf,
        score_thr=args.score_thr,
        max_instances=args.max_instances,
        crop_hw=(args.crop_size, args.crop_size),
        cheb_K=args.cheb_K,
        embed_dim=args.embed_dim,
        q_floor=args.q_floor,
    )

    mask_src, mask_tar = masks
    delta = perturbation
    print("L_inf (max abs):", delta.abs().max().item(), " target eps:", args.eps)
    print("mask_src mean:", mask_src.mean().item(), "mask_tar mean:", mask_tar.mean().item())
    print(f"patch={patch}, stride={stride}, nH={nH}, nW={nW}, N={N}")

    
    src_np = to_numpy_img(src_img)
    tar_np = to_numpy_img(tar_img)
    adv_np = to_numpy_img(adv_img)

    delta_mag = delta.abs().mean(dim=1, keepdim=True)
    delta_mag = delta_mag / (delta_mag.max() + 1e-8)
    delta_mag_np = delta_mag.squeeze(0).squeeze(0).cpu().numpy()

    ms_np = mask_src[0].detach().cpu().numpy()
    mt_np = mask_tar[0].detach().cpu().numpy()

    
    hsv_comp = hsv_delta_composite(adv_img, src_img, q=0.995)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 5, 1);
    plt.imshow(src_np);
    plt.axis('off');
    plt.title('Source')
    plt.subplot(1, 5, 2);
    plt.imshow(tar_np);
    plt.axis('off');
    plt.title('Target')
    plt.subplot(1, 5, 3);
    plt.imshow(adv_np);
    plt.axis('off');
    plt.title('Adversarial (projected)')
    plt.subplot(1, 5, 4);
    plt.imshow(hsv_comp);
    plt.axis('off');
    plt.title('Δ (HSV composite)')
    plt.subplot(1, 5, 5);
    plt.imshow(ms_np, cmap='gray');
    plt.axis('off');
    plt.title('Mask(src)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(4, 4))
    plt.imshow(mt_np, cmap='gray'); plt.axis('off'); plt.title('Mask(target)')
    plt.show()
    plot_paper_panel_hsv(src_img, tar_img, adv_img, eps_rgb=args.eps,
                         save_path="paper_panel_hsv.png",
                         viz_q=0.995, viz_gamma=0.5)



