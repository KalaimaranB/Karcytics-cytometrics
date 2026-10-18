"""Published work CytoMetrics' methods are built on.

Every entry was checked against Crossref (DOIs), OpenCV's own
``doc/opencv.bib`` (Meyer 1992) or arXiv (PyTorch) on 2026-10-17. Add new
entries the same way, never from memory (see EXPANSION_PLAN.md §6).
Pipelines name these by key in their ``citations`` attribute.
"""

from karcytics_sdk.plugin.citations import Citation, register

CITATIONS = (
    Citation(
        key="cpsam2025",
        authors=("Pachitariu, M.", "Rariden, M.", "Stringer, C."),
        title="Cellpose-SAM: superhuman generalization for cellular segmentation",
        year=2025,
        venue="bioRxiv",
        doi="10.1101/2025.04.28.651001",
        kind="misc",
    ),
    Citation(
        key="cellpose2021",
        authors=("Stringer, C.", "Wang, T.", "Michaelos, M.", "Pachitariu, M."),
        title="Cellpose: a generalist algorithm for cellular segmentation",
        year=2021,
        venue="Nature Methods",
        volume="18",
        pages="100–106",
        doi="10.1038/s41592-020-01018-x",
    ),
    Citation(
        key="pytorch2019",
        authors=(
            "Paszke, A.",
            "Gross, S.",
            "Massa, F.",
            "Lerer, A.",
            "Bradbury, J.",
            "Chanan, G.",
            "Killeen, T.",
            "Lin, Z.",
            "Gimelshein, N.",
            "Antiga, L.",
            "Desmaison, A.",
            "Köpf, A.",
            "Yang, E.",
            "DeVito, Z.",
            "Raison, M.",
            "Tejani, A.",
            "Chilamkurthy, S.",
            "Steiner, B.",
            "Fang, L.",
            "Bai, J.",
            "Chintala, S.",
        ),
        title="PyTorch: An Imperative Style, High-Performance Deep Learning Library",
        year=2019,
        venue="Advances in Neural Information Processing Systems 32 (NeurIPS 2019)",
        url="https://arxiv.org/abs/1912.01703",
        kind="inproceedings",
    ),
    Citation(
        key="otsu1979",
        authors=("Otsu, N.",),
        title="A Threshold Selection Method from Gray-Level Histograms",
        year=1979,
        venue="IEEE Transactions on Systems, Man, and Cybernetics",
        volume="9(1)",
        pages="62–66",
        doi="10.1109/TSMC.1979.4310076",
    ),
    Citation(
        key="meyer1992",
        authors=("Meyer, F.",),
        title="Color image segmentation",
        year=1992,
        venue="International Conference on Image Processing and its Applications",
        pages="303–306",
        url="https://www.researchgate.net/publication/279958069_Color_Image_Segmentation",
        kind="inproceedings",
    ),
    Citation(
        key="opencv2000",
        authors=("Bradski, G.",),
        title="The OpenCV Library",
        year=2000,
        venue="Dr. Dobb's Journal of Software Tools",
    ),
    Citation(
        key="skimage2014",
        authors=(
            "van der Walt, S.",
            "Schönberger, J. L.",
            "Nunez-Iglesias, J.",
            "Boulogne, F.",
            "Warner, J. D.",
            "Yager, N.",
            "Gouillart, E.",
            "Yu, T.",
        ),
        title="scikit-image: image processing in Python",
        year=2014,
        venue="PeerJ",
        volume="2",
        pages="e453",
        doi="10.7717/peerj.453",
    ),
)

register(*CITATIONS)
