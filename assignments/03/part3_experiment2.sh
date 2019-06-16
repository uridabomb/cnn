python -m hw3.experiments run-exp -n exp2_L1_K64-128-256-512 -K 64 128 256 512 -L 1 -P 1 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L2_K64-128-256-512 -K 64 128 256 512 -L 2 -P 2 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L3_K64-128-256-512 -K 64 128 256 512 -L 3 -P 3 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L4_K64-128-256-512 -K 64 128 256 512 -L 4 -P 4 -H 64 --ycn --reg 1e-4 --batches 200

python -m hw3.experiments run-exp -n exp2_L1_K64-128-256 -K 64 128 256 -L 1 -P 1 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L2_K64-128-256 -K 64 128 256 -L 2 -P 2 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L3_K64-128-256 -K 64 128 256 -L 3 -P 3 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L4_K64-128-256 -K 64 128 256 -L 4 -P 4 -H 64 --ycn --reg 1e-4 --batches 200

python -m hw3.experiments run-exp -n exp2_L1_K32-64-128 -K 32 64 128 -L 1 -P 1 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L2_K32-64-128 -K 32 64 128 -L 2 -P 2 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L3_K32-64-128 -K 32 64 128 -L 3 -P 3 -H 64 --ycn --reg 1e-4 --batches 200
python -m hw3.experiments run-exp -n exp2_L4_K32-64-128 -K 32 64 128 -L 4 -P 4 -H 64 --ycn --reg 1e-4 --batches 200