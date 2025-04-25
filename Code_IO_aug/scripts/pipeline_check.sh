pfn=${1:-data/rawcode_1k_parsed.jsonl}
ifn=${2:-data/codeio_1k_gens.jsonl}
ofn=${3:-data/codeio_1k_gens_verified.jsonl}
pythonpath=${4:-"python"}
runpath=${5:-"./temp/temp/temp"}

python ./src/check_io_pred_acc_mp.py \
--parsed_file_name $pfn \
--pred_file_name $ifn \
--res_file_name $ofn \
--batchsize 1024 \
--num_processes 24 \
--python_path $pythonpath \
--run_path $runpath

for i in {1..10}
do
echo "trial $i"

if [ $i -eq 1 ]; then
    numprocess=24
elif [ $i -eq 2 ]; then
    numprocess=16
elif [ $i -eq 8 ] || [ $i -eq 9 ] || [ $i -eq 10 ]; then
    numprocess=4
else
    numprocess=8
fi

echo "numprocess: $numprocess"

python ./src/check_io_pred_acc_mp_inplace.py \
--parsed_file_name $pfn \
--pred_file_name $ofn \
--batchsize 1024 \
--write_batchsize 16 \
--num_processes $numprocess \
--python_path $pythonpath \
--run_path $runpath
done