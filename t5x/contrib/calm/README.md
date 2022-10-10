# Confident Adaptive Language Modeling (CALM)

This repository contains overrides and configs for running the CALM T5 model in T5X, introduced in the NeurIPS 2022 paper: [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061).

CALM skips Transformer decoder layers when generating text by early exiting based on calibrated confidence measures.

This model should be paired with the Flaxformer [calm_t5](https://github.com/google/flaxformer/tree/main/flaxformer/architectures/calm_t5) architecture.

## Reference
When referring to this model, please cite this paper:

```
@inproceedings{Schuster2022CALM,
  title={Confident Adaptive Language Modeling},
  author={Tal Schuster and Adam Fisch and Jai Gupta and Mostafa Dehghani and Dara Bahri and Vinh Quang Tran and Yi Tay and Donald Metzler},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  url = {https://arxiv.org/abs/2207.07061},
  year={2022},
}
```


## Note
This is not an officially supported Google product.