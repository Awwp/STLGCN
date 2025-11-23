For specific configurations, please refer to https://github.com/kennymckormick/pyskl. 
Configure this model under the PYSKL framework.


run：   CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh configs/ntu60_xsub_3dkp/j.py 2 --validate --test-last --test-best
