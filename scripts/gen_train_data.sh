sh mkdirs.sh
echo 'generating supplement training data...'
python base_utils/gen_supplement_data.py \
                --input_img_dir ./data/any2any/train/input \
                --input_depth_dir ./data/any2any/train/depth \
                --save_img_dir ./data/supplement/train/input \
                --save_gt_dir ./data/supplement/train/target \
                --save_depth_dir ./data/supplement/train/depth

echo 'generating surface training normal...'
python base_utils/gen_surface_normals.py \
                --input_dir ./data/one2one/train/depth \
                --save_dir ./data/one2one/train/normals 

echo 'generating surface validation normal...'
python base_utils/gen_surface_normals.py \
                --input_dir ./data/one2one/validation/depth \
                --save_dir ./data/one2one/validation/normals

echo 'generating surface supplement data normal...'
python base_utils/gen_surface_normals.py \
                --input_dir ./data/supplement/train/depth \
                --save_dir ./data/supplement/train/normals

