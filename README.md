# gaussian-splatting-webgpu

This is a WebGPU implementation of the render for Gaussian Splatting paper. This is a work in progress and is designed to be for learning how to use WebGPU. 

## Step 1

Build a full accuracy render without using a GPU. This will be used as a reference to ensure correctness.

How to render a gaussian splatting in WebGPU.
- Load gaussians from a file.
- Prune gaussians outside the view frustum.
- Sort gaussians by depth.
- Splat the gaussians onto 2D slices.
- Render the gaussians using alpha blending.
