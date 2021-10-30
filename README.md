# Traffic4cast2021

## Data Preparation

After cloning this [repo](git@github.com:YichaoLu/Traffic4cast2021.git), download and extract [data](https://www.iarai.ac.at/traffic4cast/forums/forum/competition/competition-2021/) to the `data` folder, and then download and extract [pre-trained weights for the Core Competition](https://drive.google.com/file/d/1l6ggSXhYZPm7wwspbAboombgE6Y0stLn/view?usp=sharing) and [pre-trained weights for the Extended Competition](https://drive.google.com/file/d/1cDZ4mjyhlgP6dODbbbr1S6IMNFsAfFY-/view?usp=sharing) to the `model` folder:

```
|-- data
|   |-- ANTWERP
|   |-- BANGKOK
|   |-- BARCELONA
|   |-- BERLIN
|   |-- CHICAGO
|   |-- ISTANBUL
|   |-- MOSCOW
|   |-- NEWYORK
|   |-- VIENNA
|-- model
|   |-- model_1.bin
|   |-- model_2.bin
|   |-- model_3.bin
|   |-- model_4.bin
|   |-- model_5.bin
|   |-- model_6.bin
|   |-- model_7.bin
|   |-- v1_epoch_5.bin
|   |-- v2_epoch_5.bin
|   |-- v3_epoch_5.bin
|   |-- v4_epoch_5.bin
|   |-- v5_epoch_5.bin
|   |-- v6_epoch_5.bin
|   |-- v7_epoch_5.bin
|   |-- v8_epoch_5.bin
|   |-- v9_epoch_5.bin
```

## Usage

To create a submission for the [Core Challenge](https://www.iarai.ac.at/traffic4cast/competitions/t4c-2021-core-temporal/?leaderboard), run

```
cd core_competition
python inference_v1.py --submission_name model_v1 --checkpoint_path ../model/v1_epoch_5.bin
python inference_v2.py --submission_name model_v2 --checkpoint_path ../model/v2_epoch_5.bin
python inference_v3.py --submission_name model_v3 --checkpoint_path ../model/v3_epoch_5.bin
python inference_v4.py --submission_name model_v4 --checkpoint_path ../model/v4_epoch_5.bin
python inference_v5.py --submission_name model_v5 --checkpoint_path ../model/v5_epoch_5.bin
python inference_v6.py --submission_name model_v6 --checkpoint_path ../model/v6_epoch_5.bin
python inference_v7.py --submission_name model_v7 --checkpoint_path ../model/v7_epoch_5.bin
python inference_v8.py --submission_name model_v8 --checkpoint_path ../model/v8_epoch_5.bin
python inference_v9.py --submission_name model_v9 --checkpoint_path ../model/v9_epoch_5.bin
python submission.py
```

This creates a submission file named `core_submission.zip` under the `submission` folder.

To create a submission for the [Extended Challenge](https://www.iarai.ac.at/traffic4cast/competitions/t4c-2021-extended-spatiotemporal/?leaderboard), run

```
python inference.py --submission_name extended_model_v1 --checkpoint_path ../model/model_1.bin
python inference.py --submission_name extended_model_v2 --checkpoint_path ../model/model_2.bin
python inference.py --submission_name extended_model_v3 --checkpoint_path ../model/model_3.bin
python inference.py --submission_name extended_model_v4 --checkpoint_path ../model/model_4.bin
python inference.py --submission_name extended_model_v5 --checkpoint_path ../model/model_5.bin
python inference.py --submission_name extended_model_v6 --checkpoint_path ../model/model_6.bin
python inference.py --submission_name extended_model_v7 --checkpoint_path ../model/model_7.bin
python submission.py
```

This creates a submission file named `extended_submission.zip` under the `submission` folder.

## Acknowledgements
This repository is based on [NeurIPS2021-traffic4cast](https://github.com/iarai/NeurIPS2021-traffic4cast) from [the Institute of Advanced Research in Artificial Intelligence (IARAI)](http://www.iarai.ac.at).
