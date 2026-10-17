# How to Cite

CytoMetrics is built on other scientists' published work. If CytoMetrics contributes to your results, please cite the methods you used, not just the software.

## Always

- **Karcytics CytoMetrics**, version from the Hub's plugin list. <https://github.com/KalaimaranB/Karcytics-cytometrics>

## If you used AI Smart Detect (Cellpose)

- Pachitariu, M., Rariden, M. & Stringer, C. *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv (2025). [doi:10.1101/2025.04.28.651001](https://doi.org/10.1101/2025.04.28.651001)
- Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods 18, 100–106 (2021). [doi:10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)
- Paszke, A. et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems 32 (2019). [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)

## If you used Basic Threshold (Otsu)

- Otsu, N. *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics 9(1), 62–66 (1979). [doi:10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)

## If you used Watershed (Clump Splitting)

- Meyer, F. *Color image segmentation.* International Conference on Image Processing and its Applications, 303–306, IET (1992). This is the marker-based watershed variant implemented by OpenCV's `cv2.watershed`.
- Otsu (1979), above: Watershed uses Otsu's threshold to find foreground and seeds.

## Classical image processing (Otsu and Watershed)

- Bradski, G. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools (2000).

!!! note "Coming soon"
    CytoMetrics will write a ready-made methods paragraph and a BibTeX file next to every export, listing exactly the models, versions and settings you used (Expansion Plan §6).
