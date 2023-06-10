import torch
import torch.nn as nn

from .BasicBlock import ResidualBlockWithStride, ResidualBlockUpsample, DepthConvBlock


class MotionAnalysisTransform(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride(in_ch=in_ch, out_ch=out_ch, stride=2),
            DepthConvBlock(in_ch=out_ch, out_ch=out_ch),
        )
        self.enc_2 = ResidualBlockWithStride(in_ch=out_ch, out_ch=out_ch, stride=2)
        self.enc_3 = nn.Sequential(
            ResidualBlockWithStride(in_ch=out_ch, out_ch=out_ch, stride=2),
            DepthConvBlock(in_ch=out_ch, out_ch=out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        )

        self.adaptor_0 = DepthConvBlock(in_ch=out_ch, out_ch=out_ch)
        self.adaptor_1 = DepthConvBlock(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x: torch.Tensor, quant_step: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        out = self.enc_1(x)
        out = out * quant_step
        out = self.enc_2(out)
        if context is None:
            out = self.adaptor_0(out)
        else:
            out = self.adaptor_1(torch.cat([out, context], dim=1))
        return self.enc_3(out)


class MotionSynthesisTransform(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dec_1 = nn.Sequential(
            DepthConvBlock(in_ch=in_ch, out_ch=in_ch),
            ResidualBlockUpsample(in_ch=in_ch, out_ch=in_ch),
            DepthConvBlock(in_ch=in_ch, out_ch=in_ch),
            ResidualBlockUpsample(in_ch=in_ch, out_ch=in_ch),
            DepthConvBlock(in_ch=in_ch, out_ch=in_ch)
        )
        self.dec_2 = ResidualBlockUpsample(in_ch=in_ch, out_ch=in_ch)
        self.dec_3 = nn.Sequential(
            DepthConvBlock(in_ch=in_ch, out_ch=in_ch),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2)
        )

    def forward(self, x: torch.Tensor, quant_step: torch.Tensor) -> tuple:
        feature = self.dec_1(x)
        out = self.dec_2(feature) * quant_step
        mv = self.dec_3(out)
        return mv, feature


class MotionHyperEncoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MotionHyperDecoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch * 4, kernel_size=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class MotionCompression(nn.Module):
    def __init__(self, mv_ch: int = 2, N: int = 64, M: int = 64):
        super().__init__()
        self.g_a = MotionAnalysisTransform(in_ch=mv_ch, out_ch=N)
        self.g_s = MotionSynthesisTransform(in_ch=N, out_ch=mv_ch)
        self.h_a = MotionHyperEncoder(in_ch=N, out_ch=M)
        self.h_s = MotionHyperDecoder(in_ch=M, out_ch=N)

        self.mv_y_prior_fusion_adaptor_0 = DepthConvBlock(in_ch=mv_ch, out_ch=mv_ch * 2)
        self.mv_y_prior_fusion_adaptor_1 = DepthConvBlock(in_ch=mv_ch * 2, out_ch=mv_ch * 2)

        self.mv_y_prior_fusion = nn.Sequential(
            DepthConvBlock(in_ch=mv_ch * 2, out_ch=mv_ch * 3),
            DepthConvBlock(in_ch=mv_ch * 3, out_ch=mv_ch * 3)
        )

        self.mv_y_spatial_prior_adaptor_1 = nn.Conv2d(in_channels=mv_ch * 4, out_channels=mv_ch * 3, kernel_size=1)
        self.mv_y_spatial_prior_adaptor_2 = nn.Conv2d(in_channels=mv_ch * 4, out_channels=mv_ch * 3, kernel_size=1)
        self.mv_y_spatial_prior_adaptor_3 = nn.Conv2d(in_channels=mv_ch * 4, out_channels=mv_ch * 3, kernel_size=1)

        self.mv_y_spatial_prior = nn.Sequential(
            DepthConvBlock(in_ch=mv_ch * 3, out_ch=mv_ch * 3),
            DepthConvBlock(in_ch=mv_ch * 3, out_ch=mv_ch * 3),
            DepthConvBlock(in_ch=mv_ch * 3, out_ch=mv_ch * 2)
        )


        