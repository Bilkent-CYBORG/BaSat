# BayOPy: one Overleaf project, two guides

Upload all contents of this folder, or the supplied overlaf.zip, into one Overleaf project.
Select **pdfLaTeX** as the compiler. In the project settings, select the main document:

- **main1.tex**: BayOPy last version user guide, including saved terminal outputs.
- **main2.tex**: BayOPy user guide, with explanations, editable code and output figures only.

Both retain the same sections and environment-adaptation appendix. Both use the current notebook and its refreshed saved-result figures. Optimization was not rerun for this export.

Edit prose in the selected .tex file, code in code/*.py, and shared figures in figures/. Main1 reads outputs/*.txt; main2 does not. Code files are printed, not executed by LaTeX. No shell escape or Python installation is needed to compile the documents. Editing code does not regenerate results; use the notebook for that.

The experiment data and full result JSONs remain in the parent project's data/ and results/ directories. They are not required by Overleaf. result_provenance.json records the saved results used for the figures; execution_manifest.json and execution_requirements.txt retain execution provenance.

Local compilation: `latexmk -pdf main1.tex` or `latexmk -pdf main2.tex`.
