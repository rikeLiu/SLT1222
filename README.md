Dependencies: torch, transformers, datasets, evaluate

Train a model with

```shell
python train.py --data_path dataset/ph14_test/ --src_lang gloss --tgt_lang de --model_name Helsinki-NLP/opus-mt-en-de --experiment_name ph14 --lr 1e-4 --do_train

python train.py --data_path dataset/aslg_test/ --src_lang gloss --tgt_lang en --model_name Helsinki-NLP/opus-mt-de-en --experiment_name aslg --lr 1e-4 --do_train 
```

Evaluate a model with

```shell
python train.py --data_path dataset/ph14_test/ --model_name Helsinki-NLP/opus-mt-en-de  --experiment_name ph14 --lr 1e-4 --do_predict --src_lang gloss --tgt_lang de

python train.py --data_path dataset/aslg_test/ --model_name Helsinki-NLP/opus-mt-de-en  --experiment_name aslg --lr 1e-4 --do_predict --src_lang gloss --tgt_lang en
```