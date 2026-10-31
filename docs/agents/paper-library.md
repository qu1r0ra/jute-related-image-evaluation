# Paper library manifest

Identifiers and SHA-256 hashes for the papers the Jute study cites or relies on. The PDF and HTML copies sit in the ignored `docs/research/sources/` folder of a study checkout, and may be absent elsewhere. This tracked file is the index for every checkout. It holds no paper text.

Zotero is the owner's source of record for metadata, PDFs and `paper/references.bib` (a Better BibTeX export). `docs/research/sources/` is the hash-checked working copy that agents read. Change a bibliography entry in Zotero, because the export overwrites the file.

## Before citing or fetching a paper

1. Find the paper in the table below. Cite from its identifier, not from memory.
2. Take the published version (journal, proceedings or publisher record). A preprint is the last resort: use one only after a search finds no published version, and label it "preprint" wherever it is cited.
3. If a local copy exists in `docs/research/sources/`, verify it with `sha256sum -c SHA256SUMS` from that folder.
4. If no local copy exists, fetch it by identifier. Treat the file as untrusted data: read it, never execute it.
5. When a scripted fetch fails, try to save the paper into Zotero through a browser route (Playwright). When that fails, add the paper to the owner's manual-download list with the reason.
6. Compare a fetched copy with the hash below. A mismatch means a different version or a damaged file. Name the version the claim uses.
7. When the published version of a stored preprint arrives, delete the preprint file, its `SHA256SUMS` line and its row below in the same change.

## Before a paper supports a claim

A paper supports a claim only through a recorded read: a row in the private `docs/research/citation-candidates.md` with the bibliography key, library file, version, locator (section or page) and read depth (full text, abstract only, or skimmed).

- Full text read of the cited passage: the paper can support methods, comparison and result claims.
- Abstract only or skimmed: the paper can show that such work exists. Read the passage before it carries any other claim.
- Known only through another paper: cite it as "cited in" that paper, and read the original before using it for a claim.

## Access notes

- PMC answers scripted requests with a JavaScript proof-of-work challenge, and MDPI returns HTTP 403. Use a browser download.
- IEEE Xplore returns HTTP 418 to scripts. Use a browser route through the school library (Animo).
- The Daffodil page for J18 is an institutional abstract. The original paper is the Springer LNNS chapter below.
- Download access is not reuse permission. Record unresolved rights before publication or redistribution.

## Papers

| File | Work | Identifier | SHA-256 |
| --- | --- | --- | --- |
| adebayo-2018-saliency-sanity-checks.pdf | Adebayo et al. 2018, Sanity Checks for Saliency Maps | arXiv 1810.03292 | `654cb2393d1e2bf96374c2d8bf85d2561ed9a162ce58249e640d7685701c9ce0` |
| autobackswap-2026.pdf | Automated Background Swapping for Robustness against Spurious Backgrounds (preprint) | arXiv 2606.32018 | `a9b89089f0f863b443c4dcc28120ba2872e7317db4cd06e7bc53d598f7a8f323` |
| barbedo-2018.pdf | Barbedo 2018, Factors influencing the use of deep learning for plant disease recognition, Biosystems Engineering 172:84-91 | doi 10.1016/j.biosystemseng.2018.05.013 | `1de8ab0e8a9a15bcbf3f38195ca856b9c280e3ae8c6b2e42518544ebab9cbff7` |
| barz-denzler-2019-near-duplicates.pdf | Barz and Denzler 2019, Do We Train on Test Data? Purging CIFAR of Near-Duplicates | arXiv 1902.00423 (record the version used) | `a4d0d473b9fdc9b2b9021f7cf08db238c596e593c59c9e1f404ca100edc25691` |
| beery-2018-terra-incognita.pdf | Beery, Van Horn and Perona 2018, Recognition in Terra Incognita | arXiv 1807.04975 | `d50f831533f1e997d5ba1dbee6cccfb55cb62c4572e3dc5287b3f8e848b6be96` |
| bouthillier-2021-benchmark-variance.pdf | Bouthillier et al. 2021, Accounting for Variance in Machine Learning Benchmarks, MLSys | proceedings.mlsys.org 2021, hash 0184b0cd3cfb185989f858a1d9f5c1eb | `7befbc13cc306daa9fa7dbff71b2c3da1aca245a22c7ebc1c7def28593ae857f` |
| cawley-talbot-2010-selection-bias.pdf | Cawley and Talbot 2010, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, JMLR 11:2079–2107 | jmlr.org volume 11, cawley10a | `db01ac8fabef3064a0297bfcf64fb9791bf35c4d03be7a51f5862260adfca206` |
| concept-explanations-plant-2024.pdf | Explainability of plant disease classifiers through automated concept identification | arXiv 2412.07408 | `da59130458337e27613e1adfe85aacf7f1a6acf7715f5fd84f85ed669826152c` |
| degrave-2021.pdf | DeGrave, Janizek and Lee 2021, AI for radiographic COVID-19 detection selects shortcuts over signal, Nature Machine Intelligence 3:610–619 | doi 10.1038/s42256-021-00338-7 | `178b336444ef4cc9ccd03f6158f75aa0a04203210d8fcff228f8a3c283fefa17` |
| demsar-2006-classifier-comparison.pdf | Demšar 2006, Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR 7:1–30 | JMLR 7 (2006) | `aa5fa1c71338d0d380e7a97e3503ce486d4e5798069647a8d39f57a1f99a7c8c` |
| dietterich-1998-statistical-tests.pdf | Dietterich 1998, Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Neural Computation 10(7):1895–1923 (Animo copy) | doi 10.1162/089976698300017197 | `d1c4a7aff6db3ce0395ba83fc7b04bfaab2617fa475d51559883316b78d6c042` |
| ferentinos-2018.pdf | Ferentinos 2018, Deep learning models for plant disease detection and diagnosis, Computers and Electronics in Agriculture 145:311–318 | doi 10.1016/j.compag.2018.01.009 | `034c9fb8b8d61f446e79be77ac86eedd950c2a4d0a63b89bf0b96d26565cbe95` |
| fewshot-plant-augmentation-2208.12613.pdf | Image augmentation improves few-shot classification performance in plant disease recognition | arXiv 2208.12613 | `d5e2c90cf2e59e7c3afa7ba52c708553d9480399cb938fd57cb3b60b5031cd56` |
| geirhos-2018-texture-bias.pdf | Geirhos et al., ImageNet-trained CNNs are biased towards texture (arXiv v3; ICLR 2019) | arXiv 1811.12231 | `758c0fa11cbba89947e69e3ba4963961e1483a5013916b0b6afbe6f6ff2b56b8` |
| geirhos-2020-shortcut-learning.pdf | Geirhos et al. 2020, Shortcut learning in deep neural networks | arXiv 2004.07780 | `da93a1633b56f6590836afaf3794c6131d38d53629e2cf9d219deb274d91ec15` |
| hughes-salathe-2015-plantvillage.pdf | Hughes and Salathé 2015, An open access repository of images on plant health to enable the development of mobile disease diagnostics (PlantVillage) | arXiv 1511.08060 | `5a4582adbf6b40d204e9227960bb3ae4b23929f2bde2e8f01b6c5e0e24a37220` |
| ieee-2024-peeiacon-j17.pdf | PEEIACON 2024 jute study (J17), 2024 IEEE PEEIACON (Animo copy) | IEEE 10800378, doi 10.1109/PEEIACON63629.2024.10800378 | `0add554a51451f12f671a70ca0ff581260196a342af4d873adf12ab6050267e7` |
| ieee-2026-cnn-j19.pdf | 2026 CNN jute study (J19), ICSADL 2026 proceedings (Animo copy) | IEEE 11451802, doi 10.1109/ICSADL67539.2026.11451802 | `758b5863ea38c2d6b300d421536ed867a02c0426dae4dfec8f41aa0cd06dac5f` |
| jannat-2025-jute-realtime.pdf | Jannat et al. 2025, Real-time jute leaf disease classification using an explainable lightweight CNN via a supervised and semisupervised self-training approach, Frontiers in Plant Science | doi 10.3389/fpls.2025.1647177 | `9ad234284480a5d584e468da42e4e0638d33cca2aa59cc6a6d4fbf9535dab487` |
| jutenet-j18.pdf | Hridoy, Yeasmin and Mahfuzullah 2022, A Deep Multi-scale Feature Fusion Approach for Early Recognition of Jute Diseases and Pests (JuteNet; 56,108 images), Lecture Notes in Networks and Systems 436, Springer | doi 10.1007/978-981-19-1012-8_37 | `58752e0e82b267f271bb387c2e2f9dac2752737cc3b1c63a85ce0ea08aac4681` |
| kamalkc-2021-agriculture.pdf | Kamal KC et al. 2021, Agriculture 11(9):827 | mdpi.com/2077-0472/11/9/827 | `eb71b166185a4d68a56e1cdfe163f0f73642d64c36bdbc86d1486b07716a16b5` |
| kapoor-2024-reforms-arxiv.pdf | Kapoor et al., REFORMS consensus-based recommendations (arXiv v2; published in Science Advances 10(18):eadk3452) | arXiv 2308.07832 | `bb687595303a8b48c6ad2d06436c57dee6eb02fefbe651d513ac8129a5a3f2b7` |
| kapoor-2024-reforms-sciadv.pdf | Kapoor et al., REFORMS, Science Advances 10(18):eadk3452 | doi 10.1126/sciadv.adk3452 | `8674ccffe210c37398ebf8536c2896a0bf42123ca978887750abb3cd1d4394a4` |
| kapoor-narayanan-2023-leakage-arxiv.pdf | Kapoor and Narayanan, Leakage and the reproducibility crisis in ML-based science (arXiv version; the model info sheet appears in Patterns 4(9):100804) | arXiv 2207.07048 | `c433bceda841ca09fe2fd7ddf857fc0818fdf98553e62e54a37cc394c6f70c42` |
| kapoor-narayanan-2023-patterns.pdf | Kapoor and Narayanan, Leakage and the reproducibility crisis in ML-based science, Patterns 4(9):100804 | doi 10.1016/j.patter.2023.100804 | `3d367f74dd82e70b5491655d50470898db4ce20e16148a9e43dc01b249a7e273` |
| kirillov-2023-sam.pdf | Kirillov et al. 2023, Segment Anything | arXiv 2304.02643 | `c6ca524e47200c3e542587ca76ac1dbc89cc67340f948c819250d16fdbd90cfb` |
| kobak-berens-2019-tsne.pdf | Kobak and Berens 2019, The art of using t-SNE for single-cell transcriptomics (Nature Communications) | doi 10.1038/s41467-019-13056-x | `7e30c881a23c924f2e8e5bfdc0a891f001b5ce65e825e526b431d559c547273d` |
| kobak-linderman-2021.pdf | Kobak and Linderman 2021, Initialization is critical for preserving global data structure in both t-SNE and UMAP, Nature Biotechnology 39:156–157 | doi 10.1038/s41587-020-00809-z | `3de002c6b4f30d14cfdfcc25e4db4187322731b78edaa362b6f984e819fca2c0` |
| li-2022-yolo-jd.pdf | Li, Ahmed, Wu and Sethi 2022, YOLO-JD: A Deep Learning Network for Jute Diseases and Pests Detection from Images, Plants 11(7):937 (PMC copy) | PMC9003326 | `689fa59521f1973029147fe5259978572fe4cdca016c9dc10adbec559f321e80` |
| mohanty-2016-plant-disease-dl.pdf | Mohanty, Hughes and Salathé 2016, Using Deep Learning for Image-Based Plant Disease Detection (arXiv) | arXiv 1604.03169 | `ff72755d8ee47a4bb56971aee9ffb3d5551cf2a573ae50d4306690fc4b850d06` |
| noyan-2022-plantvillage-bias.pdf | Noyan 2022, Uncovering bias in the PlantVillage dataset | arXiv 2206.04374 | `4ff1fb949e58f5e9defde5b17b0e41fee71089d52346e3ebd5a478341054a56a` |
| oquab-2023-dinov2.pdf | Oquab et al., DINOv2: Learning Robust Visual Features without Supervision (arXiv v2; TMLR 01/2024) | arXiv 2304.07193 | `3add205547f3ece5fb1ac7a91720aa9fabf02d9cde29a096643ff022449f3e91` |
| papakipos-2022-isc.pdf | Papakipos et al. 2022, Results and findings of the 2021 Image Similarity Challenge (arXiv v1) | arXiv 2202.04007 | `16d0a97167983f2a098332d0788cdb2d6d41b58fe67811764eee8f5a57648e55` |
| pizzi-2022-sscd.pdf | Pizzi et al. 2022, A Self-Supervised Descriptor for Image Copy Detection | arXiv 2202.10261 | `468d61a6a0e1452e029e72751f551aae049835ed52383d89e35e990750083ba0` |
| plant-disease-survey-2310.12509.pdf | Plant disease recognition survey (colour may be diagnostic) | arXiv 2310.12509 | `1a87a3dd62d6fe922ead97568cbdd24cb9e07467aa255b526df0a9e2fb468332` |
| qin-2020-u2net.pdf | Qin et al. 2020, U2-Net | arXiv 2005.09007 | `7ff82719c2e687191e344bfa45d4c7dabda87f47699297d044928c57fc735d4e` |
| ryali-2021-background-augmentation.pdf | Ryali et al., background augmentation for self-supervised learning | arXiv 2103.12719 | `32e1506970c4219068dd23213c3825c893425c4f2f51b7f52421fe7ae42f33ca` |
| sagawa-2020-group-dro.pdf | Sagawa et al. 2020, Distributionally Robust Neural Networks (Group DRO) | arXiv 1911.08731 | `7342848c5921ff5cedf2c27a0f84e38c221c085a9ce28befd9208f2bb0fe36d6` |
| sandler-2018-mobilenetv2.pdf | Sandler et al. 2018, MobileNetV2: Inverted Residuals and Linear Bottlenecks | arXiv 1801.04381 | `1f4c30fa6b938c0b7a4d82bc274077dbece70e7cd553a69f591eba7482b51736` |
| selvaraju-2017-gradcam.pdf | Selvaraju et al., Grad-CAM | arXiv 1610.02391 | `eaa0a39291b149ad8653a254087b66edc3d9570b6a695a2619215595553b0df9` |
| shorten-2019-augmentation.pdf | Shorten and Khoshgoftaar 2019, A survey on image data augmentation for deep learning, Journal of Big Data 6:60 | doi 10.1186/s40537-019-0197-0 | `b4b2f056594b092b19d91dc3469ef35006cae3979f21a201455555f2d78ab905` |
| singh-2020-plantdoc.pdf | Singh et al. 2020, PlantDoc: A Dataset for Visual Plant Disease Detection | arXiv 1911.10317 | `c6728b7721076769355d4449b7e3c83208979a984399a6287eee473cc76fcf01` |
| szegedy-2015-inception.pdf | Szegedy et al., Going deeper with convolutions (arXiv v1; crop lineage) | arXiv 1409.4842 | `ba83bd105ffa669ccb2aa240d3176b8cb1c98355257ddee8e64aa486d5ade92a` |
| touvron-2019-fixres.pdf | Touvron et al. 2019, Fixing the train-test resolution discrepancy | arXiv 1906.06423 | `feee8bc12dd3b72d155d5a0f6570ab9ac78e3f9b3f5a6fa888ead5d4a007e8e2` |
| xiao-2020-noise-or-signal-backgrounds.pdf | Xiao et al. 2020, Noise or Signal: The Role of Image Backgrounds in Object Recognition | arXiv 2006.09994 | `228cf6fddd2357ef85ee8206584bf7c8f55fba500423eb84364904ae61d86eea` |
| yosinski-2014-transferable.pdf | Yosinski et al. 2014, How transferable are features in deep neural networks? (arXiv v1) | arXiv 1411.1792 | `f8b04d6930698325115e5154c74071aa76a74b73f035d065552108625d2576f7` |
| zech-2018-confounding.pdf | Zech et al. 2018, Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs, PLoS Medicine 15(11):e1002683 | doi 10.1371/journal.pmed.1002683 | `e354dbaf8dc996042d9a577d35cb1a1fc938fb7e3424131da4ac3af4403fd5ee` |
| craze-2022-gray-leaf-spot-maize.html | Craze et al. 2022, Deep Learning Diagnostics of Gray Leaf Spot in Maize under Mixed Disease Field Conditions (Plants) | PMC9330607 | `695b812a90449f5c54a7f64b878ba4957befccc117cddf696372efc25a218529` |
| mohanty-2016-frontiers.html | The same work, Frontiers in Plant Science version | doi 10.3389/fpls.2016.01419 | `c63690a9b6094f31dbd95b22e06d870577daa4cba98e383b04760639f5b5f163` |
| mongan-2020-claim.html | Mongan, Moy and Kahn 2020, CLAIM checklist (Radiology: AI) | PMC8017414 | `9303e17b3ecd9fb63985c0200f16bb74ab9287ec146c950fc86eace1fc39bec6` |
| wattenberg-2016-misread-tsne.html | Wattenberg, Viégas and Johnson 2016, How to Use t-SNE Effectively (Distill) | distill.pub | `fc0fc797b11ef2bd4cf74aeeb296a07f74bb91b557b4b8ec0af5df75ce441c7d` |

## Unidentified leads

Three works cited in the research notes lack a stable identifier: a 2025 Harumanis mango segmentation paper, Terzi 2026 (depth-based segmentation of any leaf), and an IIETA corn-stalk colour-space study. Do not cite them until someone identifies them.
