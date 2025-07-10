export PYTHONPATH=/Share8/zhuzhanshi/CauAug/mp:$PYTHONPATH
source activate CL
cd /Share8/zhuzhanshi/CauAug/
clear

# baseline
# joint training
python main.py --dataset prostate --approach joint --epochs 50 --experiment-name prostate-joint-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach joint --epochs 50 --experiment-name mm-i-joint-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach joint --epochs 50 --experiment-name mm-o-joint-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach joint --epochs 50 --experiment-name mm-r-joint-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach joint --epochs 50 --experiment-name hippocampus-joint-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach joint --epochs 50 --experiment-name optic-i-joint-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach joint --epochs 50 --experiment-name optic-o-joint-unet --backbone unet --device-ids 5

# sequential training
python main.py --dataset prostate --approach seq --epochs 50 --experiment-name prostate-seq-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach seq --epochs 50 --experiment-name mm-i-seq-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach seq --epochs 50 --experiment-name mm-o-seq-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach seq --epochs 50 --experiment-name mm-r-seq-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach seq --epochs 50 --experiment-name hippocampus-seq-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach seq --epochs 50 --experiment-name optic-i-seq-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach seq --epochs 50 --experiment-name optic-o-seq-unet --backbone unet --device-ids 5

# ablation
# kd
python main.py --dataset prostate --approach cauaug --epochs 50 --experiment-name prostate-kd-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach cauaug --epochs 50 --experiment-name mm-i-kd-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach cauaug --epochs 50 --experiment-name mm-o-kd-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach cauaug --epochs 50 --experiment-name mm-r-kd-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach cauaug --epochs 50 --experiment-name hippocampus-kd-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach cauaug --epochs 50 --experiment-name optic-i-kd-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach cauaug --epochs 50 --experiment-name optic-o-kd-unet --backbone unet --device-ids 5

#kd + anaaug
python main.py --dataset prostate --approach cauaug --epochs 50 --experiment-name prostate-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset mm --target_class i --approach cauaug --epochs 50 --experiment-name mm-i-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset mm --target_class o --approach cauaug --epochs 50 --experiment-name mm-o-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset mm --target_class r --approach cauaug --epochs 50 --experiment-name mm-r-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset hippocampus --approach cauaug --epochs 50 --experiment-name hippocampus-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset optic --target_class i --approach cauaug --epochs 50 --experiment-name optic-i-ana-unet --backbone unet --device-ids 5 --AnaAug
python main.py --dataset optic --target_class o --approach cauaug --epochs 50 --experiment-name optic-o-ana-unet --backbone unet --device-ids 5 --AnaAug

#kd + domaug
python main.py --dataset prostate --approach cauaug --epochs 50 --experiment-name prostate-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset mm --target_class i --approach cauaug --epochs 50 --experiment-name mm-i-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset mm --target_class o --approach cauaug --epochs 50 --experiment-name mm-o-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset mm --target_class r --approach cauaug --epochs 50 --experiment-name mm-r-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset hippocampus --approach cauaug --epochs 50 --experiment-name hippocampus-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset optic --target_class i --approach cauaug --epochs 50 --experiment-name optic-i-dom-unet --backbone unet --device-ids 5 --DomAug
python main.py --dataset optic --target_class o --approach cauaug --epochs 50 --experiment-name optic-o-dom-unet --backbone unet --device-ids 5 --DomAug

#kd + domaug + anaaug = cauaug
python main.py --dataset prostate --approach cauaug --epochs 50 --experiment-name prostate-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset mm --target_class i --approach cauaug --epochs 50 --experiment-name mm-i-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset mm --target_class o --approach cauaug --epochs 50 --experiment-name mm-o-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset mm --target_class r --approach cauaug --epochs 50 --experiment-name mm-r-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset hippocampus --approach cauaug --epochs 50 --experiment-name hippocampus-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset optic --target_class i --approach cauaug --epochs 50 --experiment-name optic-i-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug
python main.py --dataset optic --target_class o --approach cauaug --epochs 50 --experiment-name optic-o-cauaug-unet --backbone unet --device-ids 5 --DomAug --AnaAug

# compared
# ted gudf=true
python main.py --dataset prostate --approach ted --epochs 50 --experiment-name prostate-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset mm --target_class i --approach ted --epochs 50 --experiment-name mm-i-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset mm --target_class o --approach ted --epochs 50 --experiment-name mm-o-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset mm --target_class r --approach ted --epochs 50 --experiment-name mm-r-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset hippocampus --approach ted --epochs 50 --experiment-name hippocampus-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset optic --target_class i --approach ted --epochs 50 --experiment-name optic-i-tedgugf-unet --backbone unet --device-ids 5 --gugf
python main.py --dataset optic --target_class o --approach ted --epochs 50 --experiment-name optic-o-tedgugf-unet --backbone unet --device-ids 5 --gugf

# ewc
python main.py --dataset prostate --approach ewc --epochs 50 --experiment-name prostate-ewc-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach ewc --epochs 50 --experiment-name mm-i-ewc-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach ewc --epochs 50 --experiment-name mm-o-ewc-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach ewc --epochs 50 --experiment-name mm-r-ewc-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach ewc --epochs 50 --experiment-name hippocampus-ewc-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach ewc --epochs 50 --experiment-name optic-i-ewc-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach ewc --epochs 50 --experiment-name optic-o-ewc-unet --backbone unet --device-ids 5

# mas
python main.py --dataset prostate --approach mas --epochs 50 --experiment-name prostate-mas-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach mas --epochs 50 --experiment-name mm-i-mas-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach mas --epochs 50 --experiment-name mm-o-mas-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach mas --epochs 50 --experiment-name mm-r-mas-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach mas --epochs 50 --experiment-name hippocampus-mas-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach mas --epochs 50 --experiment-name optic-i-mas-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach mas --epochs 50 --experiment-name optic-o-mas-unet --backbone unet --device-ids 5

# mib
python main.py --dataset prostate --approach mib --epochs 50 --experiment-name prostate-mib-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach mib --epochs 50 --experiment-name mm-i-mib-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach mib --epochs 50 --experiment-name mm-o-mib-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach mib --epochs 50 --experiment-name mm-r-mib-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach mib --epochs 50 --experiment-name hippocampus-mib-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach mib --epochs 50 --experiment-name optic-i-mib-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach mib --epochs 50 --experiment-name optic-o-mib-unet --backbone unet --device-ids 5

# plop
python main.py --dataset prostate --approach plop --epochs 50 --experiment-name prostate-plop-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach plop --epochs 50 --experiment-name mm-i-plop-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach plop --epochs 50 --experiment-name mm-o-plop-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach plop --epochs 50 --experiment-name mm-r-plop-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach plop --epochs 50 --experiment-name hippocampus-plop-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach plop --epochs 50 --experiment-name optic-i-plop-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach plop --epochs 50 --experiment-name optic-o-plop-unet --backbone unet --device-ids 5

# refresh
python main.py --dataset prostate --approach refresh --epochs 50 --experiment-name prostate-refresh-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach refresh --epochs 50 --experiment-name mm-i-refresh-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach refresh --epochs 50 --experiment-name mm-o-refresh-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach refresh --epochs 50 --experiment-name mm-r-refresh-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach refresh --epochs 50 --experiment-name hippocampus-refresh-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach refresh --epochs 50 --experiment-name optic-i-refresh-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach refresh --epochs 50 --experiment-name optic-o-refresh-unet --backbone unet --device-ids 5

# aug_type
# standard
python main.py --dataset prostate --approach kd --epochs 50 --experiment-name prostate-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset mm --target_class i --approach kd --epochs 50 --experiment-name mm-i-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset mm --target_class o --approach kd --epochs 50 --experiment-name mm-o-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset mm --target_class r --approach kd --epochs 50 --experiment-name mm-r-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset hippocampus --approach kd --epochs 50 --experiment-name hippocampus-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset optic --target_class i --approach kd --epochs 50 --experiment-name optic-i-standard-unet --backbone unet --device-ids 5 --aug_type standard
python main.py --dataset optic --target_class o --approach kd --epochs 50 --experiment-name optic-o-standard-unet --backbone unet --device-ids 5 --aug_type standard

# blur
python main.py --dataset prostate --approach kd --epochs 50 --experiment-name prostate-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset mm --target_class i --approach kd --epochs 50 --experiment-name mm-i-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset mm --target_class o --approach kd --epochs 50 --experiment-name mm-o-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset mm --target_class r --approach kd --epochs 50 --experiment-name mm-r-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset hippocampus --approach kd --epochs 50 --experiment-name hippocampus-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset optic --target_class i --approach kd --epochs 50 --experiment-name optic-i-blur-unet --backbone unet --device-ids 5 --aug_type blur
python main.py --dataset optic --target_class o --approach kd --epochs 50 --experiment-name optic-o-blur-unet --backbone unet --device-ids 5 --aug_type blur

# drfr
python main.py --dataset prostate --approach kd --epochs 50 --experiment-name prostate-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset mm --target_class i --approach kd --epochs 50 --experiment-name mm-i-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset mm --target_class o --approach kd --epochs 50 --experiment-name mm-o-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset mm --target_class r --approach kd --epochs 50 --experiment-name mm-r-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset hippocampus --approach kd --epochs 50 --experiment-name hippocampus-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset optic --target_class i --approach kd --epochs 50 --experiment-name optic-i-drfr-unet --backbone unet --device-ids 5 --aug_type drfr
python main.py --dataset optic --target_class o --approach kd --epochs 50 --experiment-name optic-o-drfr-unet --backbone unet --device-ids 5 --aug_type drfr

# ginipa
python main.py --dataset prostate --approach ginipa --epochs 50 --experiment-name prostate-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach ginipa --epochs 50 --experiment-name mm-i-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach ginipa --epochs 50 --experiment-name mm-o-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach ginipa --epochs 50 --experiment-name mm-r-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach ginipa --epochs 50 --experiment-name hippocampus-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach ginipa --epochs 50 --experiment-name optic-i-ginipa-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach ginipa --epochs 50 --experiment-name optic-o-ginipa-unet --backbone unet --device-ids 5

# gin
python main.py --dataset prostate --approach gin --epochs 50 --experiment-name prostate-gin-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class i --approach gin --epochs 50 --experiment-name mm-i-gin-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class o --approach gin --epochs 50 --experiment-name mm-o-gin-unet --backbone unet --device-ids 5
python main.py --dataset mm --target_class r --approach gin --epochs 50 --experiment-name mm-r-gin-unet --backbone unet --device-ids 5
python main.py --dataset hippocampus --approach gin --epochs 50 --experiment-name hippocampus-gin-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class i --approach gin --epochs 50 --experiment-name optic-i-gin-unet --backbone unet --device-ids 5
python main.py --dataset optic --target_class o --approach gin --epochs 50 --experiment-name optic-o-gin-unet --backbone unet --device-ids 5
