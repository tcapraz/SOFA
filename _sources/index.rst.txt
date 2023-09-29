Welcome to arraylib's documentation!
===================================

`arraylib-solve` is a tool to deconvolve combinatorially pooled arrayed random mutagenesis libraries (e.g. by transposon mutagenesis). In a typical experiment generating arrayed mutagenesis libraries, first a pooled version of the library is created and arrayed on a grid of well plates. To infer the identities of each mutant on the well plate, wells are pooled in combinatorial manner such that each mutant appears in a unique combination of pools. The pools are then sequenced using NGS and sequenced reads are stored in individual fastq files per pool. `arraylib-solve` deconvolves the pools and returns summaries stating the identity and location of each mutant on the original well grid. The package is based on the approach described in :cite:p:`baym2016rapid`.




.. toctree::
   :maxdepth: 2
   :caption: API:

   spFA


.. toctree::
   :maxdepth: 2
   :caption: Tutorials:
   

   notebooks/tcga_data

   

.. bibliography:: references.bib
