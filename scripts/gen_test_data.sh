sh mkdirs.sh
echo 'generating testing data normals'
python base_utils/gen_surface_normals.py \
            --input_dir ./data/test/depth \
            --save_dir ./data/test/normals