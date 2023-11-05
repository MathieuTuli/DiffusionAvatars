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

import numpy as np
import torch


class Renderer:
    def __init__(self, raster_settings, intrinsics, extrinsics, lights):
        # assert len(intrinsics) != len(extrinsics) and (len(intrinsics) != 1 or # NOQA
        #                                                len(extrinsics) != 1), \ # NOQA
        #     "Number of cameras (entries of instrinsics/extrinsics) should " + \ # NOQA
        #     "match. Currently:" + \
        #     f"{len(intrinsics)} instrinsics vs. {len(extrinsics)} extrinsics"
        self.raster_settings = raster_settings
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.lights = lights

    def process_trinsics(self, settings, keys: List[str]):
        if isinstance(self.extrinsics, dict):
            for key in keys:
                val = settings[key]
                if val.startswith("^np."):
                    # TODO this is potentially unsafe
                    exec(f"ret = {val[0:]}")
                    settings[key] = ret  # NOQA
            return [settings[key] for key in keys]
        return [[item[key] for item in settings] for key in keys]

    def process_extrinsics(self):
        return self.process_trinsics(self.extrinsics, ("dist", "elev", "azim"))

    def process_intrinsics(self):
        return self.process_trinsics(self.intrinsics, ("aspect_ration", "fov"))

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

        lights = PointLights(location=self.lights, device=device)
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftPhongShader(
                cameras=cameras,
                lights=lights
            ),
        )
        return renderer(meshes)
