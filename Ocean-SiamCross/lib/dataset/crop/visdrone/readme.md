# Preprocessing VISDRONE (train and val)


### Crop & Generate data info (20 min)

````shell
rm ./train/list.txt
rm ./val/list.txt

python parse_visdrone.py
python par_crop.py 511 16
python gen_json.py
````
