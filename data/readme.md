# Readme
Due to data copyright reasons, we only provide the all the few-shot version data in the experiment, and hide all slot labels.

Get full StandfordLU data at [link](https://atmahou.github.io/attachments/StanfordLU.zip), which contains both slot and intent labels for full data.
To use this data, please cite:
```
@inproceedings{hou2018coling,
	author    = {Yutai Hou and
	Yijia Liu and
	Wanxiang Che and
	Ting Liu},
	title     = {Sequence-to-Sequence Data Augmentation for Dialogue Language Understanding},
	booktitle = {Proc. of COLING},
	pages     = {1234--1245},
	year      = {2018},
}
```

# Data Format
few-shot/meta-episode style data example
```
{
  "domain_name": [
    {  // episode
      "support": {  // support set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],  // input sequence
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]],  // output sequence in sequence labeling task
        "labels": [["statement"], ["query"]]  // output labels in classification task
      },
      "query": {  // query set
        "seq_ins": [["we", "are", "friends", "."], ["how", "are", "you", "?"]],
        "seq_outs": [["O", "O", "O", "O"], ["O", "O", "O", "O"]],
        "labels": [["statement"], ["query"]]
      }
    },
    ...
  ],
  ...
}
```
