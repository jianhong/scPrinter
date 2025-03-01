=============
API reference
=============

An overview of scprinter API.

Some commonly used arguments in scprinter:

- `cell_grouping` : list[list[str]] | list[str] | np.ndarray
    Essentially, pseudo-bulks, specifiec by a list of the cell barcodes belong to this group, e.g.
        `['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112', 'ACAGTGGT,ACAGTGGT,ATCACGTT,BUENSS112', 'ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']`.  If you want to visualize multiple groups, you can provide a list of lists, e.g.
        `[['ACAGTGGT,ACAGTGGT,ACTTGATG,BUENSS112'] , ['ACAGTGGT,ACAGTGGT,TACTAGTC,BUENSS112', 'ACAGTGGT,ACAGTGGT,TAGTGACT,BUENSS112','ACAGTGGT,ACAGTGGT,TCCGTCTT,BUENSS112']]`.
- `group_names`: list[str] | str
    The names of the groups, e.g. `['Group1', 'Group2']` It needs to have the same length as the `cell_grouping` list.
- `save_group_names`: list[str] | str
    Some functions in scPrinter allow you to collapse or merge multiple groups into one group (specified by a list of `group_names`). In this case you can specify the `save_group_names` to name these new groups. If you don't specify it, it will use the `group_names` as the default.
    the relationship can be seens as: barcode - single cell; a list of barcode (`cell_grouping`) - pseudo-bulk (named by `group_names`); a list of pseudo-bulks (specified by `group_names`) - bigger pseudo-bulk (named by `save_group_names`)
- `sample_names`: list[str] | str
    The names of the samples or library. Only relevant when you have multiple fragments file. When provided, it will append the sample name to cell barcodes in each fragments file to avoid cell barcode collision. This is helpful for 10X scATAC-seq and multiome data where there can be multiple fragments files with the same cell barcodes.
- `save_key`: str
    If you generate footprints / TF binding score for multiple region / groups, you can specify a key to save the results in the printer object or local path. `save_key` refers to the collection of these results
- `save_path`: str
    The path to save the results. It needs to contain the file name as well, such as `/data/rzhang/modisco.h5`
- `wandb_project`: str
    The wandb project name to log the training process. If you don't want to log the training process, you can set it to `None`. But I highly recommend you to log the training process, so that you can track the training process and compare the results across different runs.
    Check https://wandb.ai/home for more details.

.. toctree::
   :maxdepth: 2

   io
   preprocessing
   tools
   plotting
   motifs
   datasets
   peak
   chromvar
   dorc
   buencolors
   utils
