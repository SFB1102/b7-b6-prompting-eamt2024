### Paper
This repo contains outputs and code for the EAMT-2024 paper: 
*Mitigating Translationese with GPT-4: Strategies and Performance*.

``@inproceedings{Kunilovskaya2024prompting,
author = {Kunilovskaya, Maria and Chowdhury, Koel Dutta and Przybyl, Heike and {Espa{\~{n}}a i Bonet}, Cristina and {Van Genabith}, Josef},
title = {{Mitigating Translationese with GPT-4: Strategies and Performance}},
booktitle = {Proceedings of the 25th Annual conference of the European Association for Machine Translation},
month = {24-27 June},
pages = {in print},
publisher = {Association for Computational Linguistics},
address = {Sheffield, UK},
year = {2024}
}``

### Overall goal, design and motivation
**Instruct GPT-4 to rewrite human translations to reduce the traces of the translation process 
and make translations less distinct from register-comparable non-translations in the target language (TL).**

Evaluation: successfully re-written translations should return lower accuracy in a binary classification 
against non-translations in the TL than the input translations before re-writing.

An emerging NLP method -- a generative LLM -- is used to validate theoretically-motivated hand-engineered 
translationese features, which inject human knowledge about the properties of translation into the LLM-powered re-writing process.
In effect, LLM is prompted to post-edit an existing human translation segment-by-segment following an instruction customised to each individual segment. 
Each instruction is formulated taking into account 
(i) the properties of the current segment, 
(ii) the target language norm (represented by average values in non-translations),
(iii) the knowledge about typical translationese deviations.

### Disclaimers
The re-writing process can generate text that deviates from non-translations along other parameters.

It is a theoretical question whether is whether translations **should be** less translated.
The property of being a translation is part of the true nature of all translated language. 
They are produced within the target language subsystem akin to a dialect, which marks them as language produced under specific communicative conditions. 
Probably, true unedited translations cannot blend seamlessly with non-translations in the target language without losing their property of being a translation, of being a faithful representative of content originally-produced in another language.

However, the amount of detectable translationese and its properties can reflect individual translation strategy. 
Translation strategy can be thought of as an interplay of 
* professional competence/experience, 
* professional norms associated with the given type of translation tasks in the given translation direction and translator community, 
* time constraints associated with a particular translation task, and more importantly,
* individual translaor's strategy, i.e. a set of consistent translation solutions reflecting the translator's preferences/habits in how the balance between accuracy and fluency is maintained.

The task of debiasing translationese (detecting and eradicating the translationese features), being solved in this project, can be useful for 
levelling out the impact of individual translator strategies and delivering an even more homogeneous output in the target language 
that is less distingushable from non-translations in the target language than before the debiasing transformation.

### Project organisation
For simplicity, this repository has minimum settings necessary to reproduce our study. 
For example, we omit the support functions that were used to re-send the failed batches and segments to GPT API again.  
We provide the output from the GPT-4 obtained originally 08-10 March 2024, as it is unlikely that the new output will be exactly the same.

The project is structured into five major steps:

| research step/folder  | description                                                                                  | 
|-----------------------|----------------------------------------------------------------------------------------------|
| 0_align/              | align [Europarl procedings](https://github.com/chozelinek/europarl)                          | 
| 1_parse_extract_feats | produce conllu annotations and use them to extract lexicogrammatical features                | 
| 2_classify1           | estimate the features and get 200 contrastive documents (100 translated, 100 non-translated) |
| 3_analysis            | exclude collinearity and estimate feature thresholds (=TL norm)                              | 
| 4_prompting           | generate individual instructions for each segment in each of the 5 modes                     | 
| 5_parse_extract2      | parse the GPT-4 output for each mode and extract features                                    | 
| 6_classify2           | run the classifier against the same contrastive non-translations                             |
| 7_evaluation          | build a multiparallel dataset (src, ht + 5 outputs), extract random 25 segments, run COMET   |



data/ folder contains various important variants of the datasets, including:
* data/raw_aligned/*.tsv.gz which were used as input to parsing (resulting in _conllu)
inputs to re-writing 
* tsv-files with raw-text bidirectional sentence-aligned EN<>RU Europarl 
* its conllu-annotated version disentangled by language (each language has translated and non-translated subset)

Each of these working folders contains scripts, res/ (possibly with named subdirs), logs/, possibly stats/ and pics/.
Where the same step is applied to input and output of re-writing transformation, 
there will be a top-level distinction between pre/ and post/ runs of the scripts. 

Included are the ultimate input _conllu_data/ and ultimate output _deliverables/
as well as temp/, README.md, requirements.txt, workflow.sh 
(which keeps track of the general order of operation and exact executed commands).


