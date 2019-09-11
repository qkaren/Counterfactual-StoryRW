# Code and data for "Counterfactual Story Reasoning and Generation"
<!--
Confidential, please do not redistribute. The code will be released under an MIT License. (What to do with this sentence?)
-->
This is the code for the following paper:

**Counterfactual Story Reasoning and Generation**
(https://arxiv.org/abs/1909.04076)

Lianhui Qin, Antoine Bosselut, Ari Holtzman, Chandra Bhagavatula, Elizabeth Clark and Yejin Choi

## Data
You can **download** the raw data from [here](https://drive.google.com/file/d/150jP5FEHqJD3TmTO_8VGdgqBftTDKn4w/view?usp=sharing) directly. 

** Supervised training data example **

```
{
  "story_id": "4fd7d150-b080-4fb1-a592-8c27fa6e1fc8",
  "premise": "Andrea wanted a picture of her jumping.",
  "initial": "She set the camera up.",
  "counterfactual": "She asked her friend to draw one.",
  "original_ending": "Then, she jumped in the air. The picture kept coming out wrong. It took twenty tries to get it right.",
  "edited_ending": [
    "Then, she jumped in the air to demonstrate how she wanted it to look.",
    "The picture kept coming out wrong.",
    "It took drawing it several times to get it right."
  ]
}
```

** Unsupervised training data example **

```
{
  "story_id": "da0e85f1-c586-4236-a8a3-ee6421c8e71d",
  "premise": "Charles' mother taught her son to carry a pre-paid cell phone.",
  "initial": "As a job seeker, Charles put his cell phone number on applications.",
  "counterfactual": "As a job seeker, Charles used his cell phone to keep his information out of employers hands.",
  "original_ending": "He needed a real cell phone, but kept up with his pre-paid cell phone. One afternoon he was in a phone interview with Apple Computers. He ran out of minutes and never reached Apple's hiring manager again."
}
```

** Dev data example (the same format as test.) **
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
