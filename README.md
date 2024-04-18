# ELF

ELF (Exponent-Less Float-point encoding) is a simple yet effective, near-errorless floating-point compression method, that transforms floating-point parameters within pre-trained models (PTMs) in such a way that the common exponent field of the transformed parameters can be completely eliminated to save storage space. ELF is embarrassingly parallel via data parallelism, achieving an extreme compression & decompression speed. 
<div align="center">
  <img width="350" alt="elf_com_decom" src="https://github.com/alexsssu/ELF/assets/21178173/1f34bb93-2fda-4baa-ab42-6be21eb32b3e">
</div>


We also developed ELVES, a compression framework integrating ELF and several other data reduction methods. ELVES uses the most effective method to compress PTMs that exhibit different patterns. 

## Publication
Everything You Always Wanted to Know About Storage Compressibility of Pre-Trained ML Models but Were Afraid to Ask (VLDB'24 to appear).

The preprint of our VLDB'24 paper can be viewed at: https://arxiv.org/abs/2402.13429.

This branch contains the source code of ELF & ELVES.

## ELF Build

## ELVES Examples
