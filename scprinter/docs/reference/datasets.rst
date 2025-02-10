========
Datasets
========
.. currentmodule:: scprinter

These functions provides an easy access to genome specific public datasets
and footprinting related datasets used in scprinter.


.. note::

    By default it will save the file to `~/.cache/scprinter_cache`
    But you can overwrite it with the environment variable `SCPRINTER_DATA`

Pretrained Models
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.dispersion_model
    datasets.TFBS_model
    datasets.TFBS_model_classI
    datasets.NucBS_model


Genome
~~~~~~

.. autosummary::
    :toctree: _autosummary

    genome.Genome
    genome.GRCh38
    genome.GRCm38
    genome.hg38
    genome.mm10
    genome.mm39

FigR datasets
~~~~~~~~~~~~~
.. autosummary::
    :toctree: _autosummary

    datasets.FigR_motifs_mouse
    datasets.FigR_motifs_human
    datasets.FigR_motifs_bagging_mouse
    datasets.FigR_motifs_bagging_human
    datasets.FigR_motifs_mouse_meme
    datasets.FigR_motifs_human_meme
    datasets.hg19TSSRanges
    datasets.hg38TSSRanges
    datasets.mm10TSSRanges


Tutorial Datasets
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: _autosummary

    datasets.BMMCTutorial
