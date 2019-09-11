# Code and data for "Counterfactual Story Reasoning and Generation"
<!--
Confidential, please do not redistribute. The code will be released under an MIT License. (What to do with this sentence?)
-->
This is the code for the following paper:

**Counterfactual Story Reasoning and Generation**
(https://arxiv.org/abs/1909.04076)

Lianhui Qin, Antoine Bosselut, Ari Holtzman, Chandra Bhagavatula, Elizabeth Clark and Yejin Choi

## Data
You can **download** the raw data from [here](https://drive.google.com/file/d/1OpLZ48OXQJWLC1GJKIMZfXuNkq9b95C_/view?usp=sharing) directly. 

```
{
  "story_id": "048f5a77-7c17-4071-8b0b-b8e43087132d",
  "premise": "Neil was visiting Limerick in Ireland.",
  "initial": "There, he saw a beautiful sight.",
  "counterfactual": "It was the ugliest city he's ever seen.",
  "original_ending": "He saw the large and lovely River Shannon! After a few minutes, he agreed with the locals. The River Shannon was beautiful.",
  "edited_endings": [
    [
      "He saw the small and lonely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was lonely."
    ],
    [
      "However, he saw the large and lovely River Shannon!",
      "After a few minutes, he agreed with the locals.",
      "The River Shannon was beautiful."
    ],
    [
      "However, he did think the large River Shannon was lovely!",
      "After a few minutes, he agreed with the locals that Limerick wasn't as ugly as he though.",
      "The River Shannon was beautiful."
    ]
  ]
}
```


## How do I cite Counterfactual Story Reasoning and Generation?
```
@inproceedings{qin-counterfactual,
    title = "Counterfactual Story Reasoning and Generation",
    author = "Qin, Lianhui and Bosselut, Antoine and Holtzman, Ari and  Bhagavatula, Chandra and  Clark, Elizabeth and Choi, Yejin",
    booktitle = "2019 Conference on Empirical Methods in Natural Language Processing.",
    month = nov,
    year = "2019",
    address = "Hongkong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/pdf/1909.04076.pdf",
}
```
