# Readme
Due to data copyright reasons, we only provide the StanfordLU data (few-shot version) in the experiment, and hide all slot labels.

Contact me to obtain full data.

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