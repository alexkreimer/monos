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
    "subcaption"
    "graphicx"
    "booktabs")
   (LaTeX-add-labels
    "eq:p2"
    "eq:pinv"
    "eq:p3"
    "eq:p4"))
 :latex)

