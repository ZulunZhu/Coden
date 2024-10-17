python setup.py build_ext --inplace&&python ogb_exp.py --dataset tmall --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --epoch 100

python setup.py build_ext --inplace&&python ogb_exp.py --dataset patent --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.2 --epoch 100