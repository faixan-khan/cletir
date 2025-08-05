# Category-level Text-to-Image Retrieval Improved: Bridging the Domain Gap with Diffusion Models and Vision Encoders

This repository contains the code for the paper ["Category-level Text-to-Image Retrieval Improved: Bridging the Domain Gap with Diffusion Models and Vision Encoders"]() published at BMVC 2025.

## Features

Pre-extracted features used in this work can be downloaded from [here]().

## Running

The provided code can be run using

```
python inf.py --help
usage: CLETIR [-h]
            [--model {clip,eva,meta,open,sig}]
            [--v_model {deit,dino}]
            [--root_dir ROOT_DIR] [--seed SEED]
            [--method {,_classdbr,_dbr}]
            [--model_path {MODEL_PATH}]
```

- Example 1: to run CLETIR on top of CLIP with class names

```
python inf.py --model clip --v_model dino --root_dir path_to_features> --model_path path_to_model
```

- Example 2: to run CLIP on top of CLIP with using descriptions

```
python inf.py --model clip --v_model dino --root_dir path_to_features> --model_path path_to_model --method _dbr
```

- Example 3: to run CLIP on top of CLIP with using descriptions and class names

```
python inf.py --model clip --v_model dino --root_dir path_to_features> --model_path path_to_model --method _classdbr
```

## Citation

```
@InProceedings{khan_2025_BMVC,
    author    = {Khan, Faizan Farooq and Stojni\'c, Vladan and Laskar, Zakaria and Elhoseiny, Mohamed and Tolias, Giorgos},
    title     = {Category-level Text-to-Image Retrieval Improved: Bridging the Domain Gap with Diffusion Models and Vision Encoders},
    booktitle = {BMVC},
    year      = {2025}
}
```