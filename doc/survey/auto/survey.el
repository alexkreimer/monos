(TeX-add-style-hook
 "survey"
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
   (LaTeX-add-bibliographies))
 :latex)

