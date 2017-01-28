(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "mathtools"
    "amssymb"
    "caption"
    "subcaption"
    "graphicx"
    "booktabs")
   (LaTeX-add-labels
    "sec:features"
    "fig:t_distrib"
    "fig:1a"
    "fig:1b"
    "fig:1c"
    "fig:feature_vectors"
    "fig:1train_t_distrib"
    "fig:2a"
    "fig:2b"
    "fig:2c"
    "fig:2d"
    "fig:1model_eval"
    "fig:2train_t_distrib"
    "fig:3a"
    "fig:3b"
    "fig:3c"
    "fig:3d"
    "fig:2model_eval"
    "fig:path_predict")
   (LaTeX-add-mathtools-DeclarePairedDelimiters
    '("abs" "")
    '("norm" "")))
 :latex)

