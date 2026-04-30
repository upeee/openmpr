# Literature

A curated reading list for multimodal product retrieval research. Entries are organized by topic, not chronologically. Contributions welcome; open a PR.

---

## Foundational vision-language models

- **CLIP (Radford et al., 2021)** — "Learning Transferable Visual Models From Natural Language Supervision." The original contrastive pretraining paper. OpenCLIP implements these checkpoints. [arXiv](https://arxiv.org/abs/2103.00020)
- **OpenCLIP (Ilharco et al., 2021)** — Open-source CLIP implementation with 190+ pretrained checkpoints. The model hub used in OpenMPR. [GitHub](https://github.com/mlfoundations/open_clip)
- **SigLIP (Zhai et al., 2023)** — "Sigmoid Loss for Language Image Pre-training." Replaces softmax contrastive loss with per-pair sigmoid, removing the dependency on negative pairs within the batch. Top SigLIP2 models achieve 77% Recall@1 on GroceryVision. [arXiv](https://arxiv.org/abs/2303.15343)
- **SigLIP2 (Tschannen et al., 2025)** — Extended SigLIP training with improved data and captioning. ViT-gopt-16-SigLIP2-384 is the top performer in OpenMPR (77.0% Recall@1 at 1.87B parameters). [arXiv](https://arxiv.org/abs/2502.14786)
- **MobileCLIP (Vasu et al., 2024)** — "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training." Knowledge distillation from ensemble teachers on DataCompDR. MobileCLIP-B (150M) outperforms 350M models trained on LAION-2B; peak semantic power density (φ=2.83) in OpenMPR. [arXiv](https://arxiv.org/abs/2311.17049)
- **ViTamin (Chen et al., 2024)** — "ViTamin: Designing Scalable Vision Models in the Vision-Language Era." Hybrid CNN-Transformer architecture. Peaks at 256px for MPR, degrades at higher resolution. [arXiv](https://arxiv.org/abs/2404.02132)
- **PE-Core / Perception Encoder (Bolya et al., 2025)** — Meta's unified encoder for vision tasks. PE-Core-L-14-336 achieves 75.8% Recall@1 at 671M parameters (98.4% of SigLIP2 peak at one-third the parameters). [arXiv](https://arxiv.org/abs/2504.13181)
- **CoCa (Yu et al., 2022)** — "CoCa: Contrastive Captioners are Image-Text Foundation Models." Combines contrastive and generative objectives. Low semantic power density on MPR due to LAION-2B pretraining noise. [arXiv](https://arxiv.org/abs/2205.01917)
- **EVA-CLIP / EVA02 (Fang et al., 2023)** — Large-scale ViT models from BAAI. EVA02-L achieves 62.4% Recall@1. [arXiv:EVA02](https://arxiv.org/abs/2303.15389)
- **CLIPA (Li et al., 2023)** — "An Inverse Scaling Law for CLIP Training." Achieves competitive accuracy with shorter training by using larger crops. ViT-L-14-CLIPA reaches 67.1% Recall@1 on DataComp-1B. [arXiv](https://arxiv.org/abs/2305.07017)

---

## Product recognition and MPR methods

- **RetailKLIP (Srivastava et al., 2023)** — Domain-adapted CLIP fine-tuned for retail product retrieval. Shows that task-specific fine-tuning improves over zero-shot but requires labeled retail data. Baseline for supervised comparisons.
- **Products-10K baseline (Bai et al., 2020)** — "Products-10K: A Large-Scale Product Recognition Dataset." 10,000 SKUs from JD.com. Used as benchmark for fine-grained product recognition. [arXiv](https://arxiv.org/abs/2008.10545)
- **RP2K (Peng et al., 2020)** — "RP2K: A Large-Scale Retail Product Dataset for Fine-Grained Image Similarity Learning." 13,000 products, retail shelf conditions. [arXiv](https://arxiv.org/abs/2006.12634)
- **GAN domain adaptation for retail (Tonioni et al., 2019)** — "Domain Adaptation for Object Detection in Retail Contexts." Transfers models across retail environments without full retraining. Illustrates the closed-set limitation that MPR addresses.
- **SGBD (Srivastava et al., 2025)** — Synthetic grocery benchmark dataset. Uses LLM-generated descriptions for retail product retrieval. Related to the synthetic description generation approach in OpenMPR.
- **MIMEX (Tur et al., 2024)** — "Exploring Fine-grained Retail Product Discrimination with Zero-shot Object Classification Using Vision-Language Models." Introduces MIMEX (28 categories, 15,277 images from micro-market environments) and benchmarks CLIP and DINOv2 ensembles for zero-shot product classification. The most directly comparable zero-shot VLM evaluation to OpenMPR's protocol, covering a different retail context. [arXiv](https://arxiv.org/abs/2409.14963)

---

## Reranking and two-stage retrieval

Information retrieval papers on re-sorting top-K shortlists. Included because the 17.5-point Recall@5-to-Recall@1 gap in MPR has an analogue in document retrieval.

- **Passage reranking with BERT (Nogueira & Cho, 2019)** — Cross-encoder reranking over top-K retrieved passages. The canonical two-stage retrieval paper in NLP. [arXiv](https://arxiv.org/abs/1901.04085)
- **ColBERT (Khattab & Zaharia, 2020)** — "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." Late interaction enables token-level matching without full cross-encoder cost. [arXiv](https://arxiv.org/abs/2004.12832)
- **SmolVLM (Marafioti et al., 2025)** — A sub-1B parameter multimodal model designed for resource-constrained inference. [HuggingFace](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

---

## Pre-training data quality

- **DataComp (Gadre et al., 2023)** — "DataComp: In Search of the Next Generation of Multimodal Datasets." Benchmark for CLIP data curation. DataComp-XL consistently outperforms LAION-2B for MPR across all architectures. [arXiv](https://arxiv.org/abs/2304.14108)
- **LAION-2B (Schuhmann et al., 2022)** — "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models." Raw-scale web data, high noise. Models trained on LAION-2B underperform DataComp-XL counterparts by up to 16% on MPR. [arXiv](https://arxiv.org/abs/2210.08402)
- **MetaCLIP (Xu et al., 2023)** — "Demystifying CLIP Data." Curates CommonCrawl to match the metadata distribution of CLIP's original training data. PE-Core uses this. [arXiv](https://arxiv.org/abs/2309.16671)
- **Chinchilla (Hoffmann et al., 2022)** — "Training Compute-Optimal Large Language Models." Scaling laws showing compute-optimal allocation between model size and training tokens. The data-quality-over-scale finding in OpenMPR mirrors the data-efficiency insight from Chinchilla. [arXiv](https://arxiv.org/abs/2203.15556)

---

## Efficiency and model selection

- **ELEVATER (Li et al., 2022)** — "ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models." Zero-shot evaluation across 20 downstream tasks. Complements OpenMPR's domain-specific evaluation. [arXiv](https://arxiv.org/abs/2204.08790)
- **Pareto efficiency analysis (Dehghani et al., 2021)** — "Efficiency in Vision Transformers." Argues for Pareto frontier analysis over single-number comparisons for model selection. Motivates the semantic power density (φ) metric in OpenMPR.
- **ViT (Dosovitskiy et al., 2020)** — "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." The Vision Transformer architecture underlying most CLIP encoders. [arXiv](https://arxiv.org/abs/2010.11929)
- **ConvNeXt (Liu et al., 2022)** — "A ConvNet for the 2020s." CNN architecture competitive with ViTs. ConvNeXt-Large-d-320 (351M) achieves 64.4% Recall@1 on LAION-2B in OpenMPR. [arXiv](https://arxiv.org/abs/2201.03545)
- **ResNet (He et al., 2016)** — "Deep Residual Learning for Image Recognition." Included in OpenCLIP as legacy baselines. Collapses on YFCC-15M (-95% Recall@1 vs WIT-400M). [arXiv](https://arxiv.org/abs/1512.03385)

---

## Challenges and competitions

- **GroceryVision Challenge (2025)** — Annual computer vision challenge with an MPR track. The evaluation dataset and protocol used in OpenMPR. 409 SKUs, 74,200 training images. [Challenge page](https://groceryvision.github.io/)
- **FGVC workshops (CVPR)** — Fine-Grained Visual Categorization workshops at CVPR cover related problems in species recognition, car models, and product variants. Relevant for cross-domain perspective on fine-grained discrimination.

---

## Contrastive learning background

- **InfoNCE (van den Oord et al., 2018)** — "Representation Learning with Contrastive Predictive Coding." The loss function underlying CLIP training. [arXiv](https://arxiv.org/abs/1807.03748)
- **PaLI / WebLI (Chen et al., 2023)** — "PaLI: A Jointly-Scaled Multilingual Language-Image Model." Introduces WebLI, the filtered web dataset used for SigLIP2 pretraining. [arXiv](https://arxiv.org/abs/2209.06794)
- **Llama 3.1 (Grattafiori et al., 2024)** — "The Llama 3 Herd of Models." The 8B-Instruct variant is used in OpenMPR to compress catalog descriptions to ≤77 tokens. [arXiv](https://arxiv.org/abs/2407.21783)

---

Entries without arXiv links are from conference proceedings or non-preprint venues. If you find a missing entry or a dead link, open an issue.
