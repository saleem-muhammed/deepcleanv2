from typing import Optional

import torch

from utils.plotting import plot_psds as plot_psds_


def get_psd(x, spectral_density: torch.nn.Module, asd: bool):
    x = x - x.mean()
    fft = torch.stft(
        x.double(),
        n_fft=spectral_density.nperseg,
        hop_length=spectral_density.nstride,
        window=spectral_density.window,
        normalized=False,
        center=False,
        return_complex=True,
    )
    fft = (fft * torch.conj(fft)).real
    stop = None if spectral_density.nperseg % 2 else -1
    fft[1:stop] *= 2
    fft *= spectral_density.scale
    if asd:
        fft = fft**0.5
    return fft.cpu().numpy()[0].T


@torch.no_grad()
def plot_psds(
    pred: torch.Tensor,
    strain: torch.Tensor,
    mask: torch.Tensor,
    spectral_density: torch.nn.Module,
    fftlength: float,
    asd: bool = True,
    fname: Optional[str] = None,
):
    mask = mask.cpu().numpy()
    cleaned = strain - pred

    cleaned = get_psd(cleaned, spectral_density, asd)[:, mask]
    raw = get_psd(strain, spectral_density, asd)[:, mask]
    pred = get_psd(pred, spectral_density, asd)[:, mask]

    freqs = torch.arange(len(mask)).cpu().numpy()
    freqs = freqs / fftlength
    freqs = freqs[mask]
    return plot_psds_(freqs, pred, raw, cleaned, asd=asd, fname=fname)
