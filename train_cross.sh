saved_result_dir='./Results/18042018'

i=0
fold_i="fold_$i"
echo "========================================================="
echo "==================== running $fold_i ====================="
echo "========================================================="
    
python train.py --word2vec_path="./Data/Word2vec/43k_word2vec.bin" --train_data_path="./Data/Train/$fold_i/train.csv" --test_data_path="./Data/Train/$fold_i/test.csv" --epochs=30 --freq_eval=20 --model_params_path="./model_params.json" --batch_size=25 --saved_result_path="$saved_result_dir/$fold_i"

i=1
fold_i="fold_$i"
echo "========================================================="
echo "==================== running $fold_i ====================="
echo "========================================================="
    
python train.py --word2vec_path="./Data/Word2vec/43k_word2vec.bin" --train_data_path="./Data/Train/$fold_i/train.csv" --test_data_path="./Data/Train/$fold_i/test.csv" --epochs=30 --freq_eval=20 --model_params_path="./model_params.json" --batch_size=25 --saved_result_path="$saved_result_dir/$fold_i"

i=2
fold_i="fold_$i"
echo "========================================================="
echo "==================== running $fold_i ====================="
echo "========================================================="
    
python train.py --word2vec_path="./Data/Word2vec/43k_word2vec.bin" --train_data_path="./Data/Train/$fold_i/train.csv" --test_data_path="./Data/Train/$fold_i/test.csv" --epochs=30 --freq_eval=20 --model_params_path="./model_params.json" --batch_size=25 --saved_result_path="$saved_result_dir/$fold_i"

i=3
fold_i="fold_$i"
echo "========================================================="
echo "==================== running $fold_i ====================="
echo "========================================================="
    
python train.py --word2vec_path="./Data/Word2vec/43k_word2vec.bin" --train_data_path="./Data/Train/$fold_i/train.csv" --test_data_path="./Data/Train/$fold_i/test.csv" --epochs=30 --freq_eval=20 --model_params_path="./model_params.json" --batch_size=25 --saved_result_path="$saved_result_dir/$fold_i"

i=4
fold_i="fold_$i"
echo "========================================================="
echo "==================== running $fold_i ====================="
echo "========================================================="
    
python train.py --word2vec_path="./Data/Word2vec/43k_word2vec.bin" --train_data_path="./Data/Train/$fold_i/train.csv" --test_data_path="./Data/Train/$fold_i/test.csv" --epochs=30 --freq_eval=20 --model_params_path="./model_params.json" --batch_size=25 --saved_result_path="$saved_result_dir/$fold_i"
