from typing import List, Dict

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)

import torch


class Renderer:
    def __init__(self, raster_settings, intrinsics, extrinsics, lights):
        if isinstance(self.extrinsics, dict):
            self.extrinsics = [self.extrinsics]
        if isinstance(self.intrinsics, dict):
            self.intrinsics = [self.intrinsics]
        assert len(intrinsics) != len(extrinsics) and (len(intrinsics) != 1 or
                                                       len(extrinsics) != 1), \
            "Number of cameras (entries of instrinsics/extrinsics) should " + \
            "match. Currently:" + \
            f"{len(intrinsics)} instrinsics vs. {len(extrinsics)} extrinsics"
        self.raster_settings = raster_settings
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.lights = lights

    def process_extrinsics(self):
        keys = ("dist", "elev", "azim")
        return [[item[key] for item in self.extrinsics] for key in keys]

    def process_intrinsics(self):
        keys = ("aspect_ratio", "fov")
        return [[item[key] for item in self.intrinsics] for key in keys]

    def render(self,
               meshes: Meshes,
               depth: bool = False,
               device: str = "cuda") -> torch.tensor:
        """
         @arguments
           - mesh: torch.tensor. Shape (V, 3)
         """
        meshes = meshes.extend(len(self.extrinsics))
        raster_settings = RasterizationSettings(
            image_size=self.raster_settings.image_size,
            blur_radius=self.raster_settings.blur_radius,
            faces_per_pixel=self.raster_settings.faces_per_pizel,
        )

        lights = PointLights(location=self.lights, device=device)

        dist, elev, azim = self.process_extrinsics()
        aspect_ratio, fov = self.process_intrinsics()
        R, T = look_at_view_transform(
            dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(
            R=R, T=T, aspect_ration=aspect_ratio, fov=fov, device=device)

        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings)

        if depth:
            return rasterizer(meshes)

        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                cameras=cameras,
                lights=lights
            ),
        )
        return renderer(meshes)
