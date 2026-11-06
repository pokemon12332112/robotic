# Third-party code and data

Upstream projects are fetched from the immutable revisions in `locks/sources.json`.
Their original LICENSE/NOTICE files remain in the fetched repositories. This kit
does not relicense upstream code, pretrained models, simulator software, or data.
Check the corresponding license before redistributing weights or Docker images.

- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [Geometric Action Model](https://github.com/cvlab-kaist/Geometric-Action-Model)
- [Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Object-understanding simulator](https://github.com/phulocnguyen/spatial-understanding)
- [Object-understanding dataset](https://huggingface.co/datasets/phulocnguyen/object-understanding)

Scene definitions and episode metadata are derived from the pinned public dataset
and preserve its provenance. They are not new independent demonstrations.
The custom adapter source was copied from the user's existing benchmark workspace.
No blanket license for the user's custom work is imposed by this packaging.
